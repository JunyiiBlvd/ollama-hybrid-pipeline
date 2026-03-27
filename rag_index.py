#!/usr/bin/env python3
# rag_index.py — Semantic embedding index for knowledge base and session memory
# v3
#
# Changes from v2:
#   - Embedding model upgraded from sentence-transformers/all-MiniLM-L6-v2
#     (22M params, 384 dims, Python-loaded) to nomic-embed-text via Ollama
#     (768 dims, retrieval-trained, server-side). Quality improvement is
#     significant for technical and domain-specific queries. Eliminates the
#     HuggingFace Hub warning and the sentence-transformers dependency.
#   - _embed_texts() replaces _get_model(). Uses Ollama /api/embed (batch)
#     instead of in-process sentence-transformers encode(). L2-normalizes
#     row-wise so cosine similarity remains a dot product.
#   - _EMBED_DIM updated 384 → 768. BREAKING CHANGE: existing pkl indexes
#     built with v2 are incompatible. Run: python rag_index.py --rebuild
#   - sentence_transformers import removed. requests added.
#
# Changes from v1:
#   - rebuild_memory() public function added — rebuilds only the memory index.
#     Called by close_thread() in thread_store.py after writing a closed-thread
#     summary so the new file is immediately queryable without a full --rebuild.
#
# WHAT THIS IS:
#   Builds and queries cosine-similarity indexes over .md files in:
#     AI/knowledge/  → knowledge.pkl
#     AI/memory/     → memory.pkl
#
#   Uses nomic-embed-text (via Ollama) for embeddings.
#   Pure numpy cosine similarity — no external vector DB required.
#   Requires: ollama pull nomic-embed-text
#
# PUBLIC API:
#   query_knowledge(prompt, top_k=1) -> list[tuple[str, str, float]]
#   query_memory(prompt, top_k=2)    -> list[tuple[str, str, float]]
#   Each returns (filename_stem, content_preview, similarity_score), best first.
#
# CLI:
#   python rag_index.py --rebuild
#   Rebuilds both indexes from scratch, prints file counts.
#
# INDEX FORMAT (pickle):
#   dict with keys: "stems", "previews", "embeddings"
#   - stems:      list[str]        — file stem names
#   - previews:   list[str]        — first 1500 chars of each file
#   - embeddings: np.ndarray       — shape (n_files, 768)

import os
import pickle
import argparse
import requests
import numpy as np
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

_VAULT        = Path(os.getenv("PIPELINE_VAULT_PATH", "./vault"))
KNOWLEDGE_DIR = _VAULT / "AI/knowledge"
MEMORY_DIR    = _VAULT / "AI/memory"
INDEX_DIR     = _VAULT / "AI/rag-index"
KNOWLEDGE_IDX = INDEX_DIR / "knowledge.pkl"
MEMORY_IDX    = INDEX_DIR / "memory.pkl"

_MODEL_NAME   = "nomic-embed-text"
_EMBED_URL    = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/embed"
_EMBED_DIM    = 768    # nomic-embed-text output dimension
_PREVIEW_CHARS = 1500


# ─── Embedding ────────────────────────────────────────────────────────────────

def _embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using nomic-embed-text via Ollama /api/embed (batch).
    Returns an (n, _EMBED_DIM) float32 array, L2-normalized row-wise so that
    cosine similarity equals the dot product.

    Raises RuntimeError if Ollama is unreachable or the model is unavailable.
    Run `ollama pull nomic-embed-text` if the model is missing.
    """
    try:
        r = requests.post(_EMBED_URL, json={"model": _MODEL_NAME, "input": texts}, timeout=120)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("[rag] Cannot connect to Ollama. Is it running?")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"[rag] Ollama embed error: {e} — run: ollama pull {_MODEL_NAME}")

    vecs = np.array(r.json()["embeddings"], dtype="float32")  # (n, dim)

    # L2-normalize row-wise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


# ─── Index Build ──────────────────────────────────────────────────────────────

def _build_index(source_dir: Path, index_path: Path) -> int:
    """
    Embed all .md files in source_dir and persist to index_path.
    Returns the number of files indexed.
    Skips unreadable files silently.
    """
    md_files = sorted(source_dir.glob("*.md"))
    if not md_files:
        _save_index(index_path, [], [], np.zeros((0, _EMBED_DIM), dtype="float32"))
        return 0

    stems, previews, texts = [], [], []

    for f in md_files:
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        stems.append(f.stem)
        preview = content[:_PREVIEW_CHARS]
        previews.append(preview)
        # Embed stem + preview for richer signal
        texts.append(f.stem.replace("-", " ").replace("_", " ") + " " + preview)

    if not texts:
        _save_index(index_path, [], [], np.zeros((0, _EMBED_DIM), dtype="float32"))
        return 0

    embeddings = _embed_texts(texts)
    _save_index(index_path, stems, previews, embeddings)
    return len(stems)


def _save_index(path: Path, stems: list, previews: list, embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"stems": stems, "previews": previews, "embeddings": embeddings}, f)


def _load_index(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Auto-build Guard ─────────────────────────────────────────────────────────

def _ensure_index(index_path: Path, source_dir: Path) -> dict | None:
    """
    Load index from disk. If missing, build it first with a warning.
    Returns None if source_dir doesn't exist or indexing yields no files.
    """
    if not index_path.exists():
        if not source_dir.exists():
            return None
        print(f"[rag] index not found — building now ({source_dir.name})")
        _build_index(source_dir, index_path)

    try:
        idx = _load_index(index_path)
    except Exception as e:
        print(f"[rag] ERROR: could not load index {index_path}: {e}")
        return None

    if len(idx["stems"]) == 0:
        return None

    # Detect stale v2 indexes (384-dim) and force rebuild
    emb = idx["embeddings"]
    if emb.ndim == 2 and emb.shape[1] != _EMBED_DIM and emb.shape[0] > 0:
        print(f"[rag] index dim mismatch ({emb.shape[1]} vs {_EMBED_DIM}) — rebuilding")
        _build_index(source_dir, index_path)
        try:
            idx = _load_index(index_path)
        except Exception:
            return None

    return idx


# ─── Cosine Query ─────────────────────────────────────────────────────────────

def _query(prompt: str, index_path: Path, source_dir: Path, top_k: int) -> list[tuple[str, str, float]]:
    """
    Embed prompt, compute cosine similarity against index, return top_k results.
    Embeddings are pre-normalized so cosine = dot product.
    Returns list of (stem, preview, score), best first.
    """
    idx = _ensure_index(index_path, source_dir)
    if idx is None:
        return []

    try:
        prompt_vec = _embed_texts([prompt])[0]  # shape (dim,)
    except Exception as e:
        print(f"[rag] query embedding failed: {e}")
        return []

    embeddings: np.ndarray = idx["embeddings"]   # shape (n, dim)
    scores = embeddings @ prompt_vec              # cosine similarity, shape (n,)

    top_k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        (idx["stems"][i], idx["previews"][i], float(scores[i]))
        for i in top_indices
    ]


# ─── Public Query API ─────────────────────────────────────────────────────────

def query_knowledge(prompt: str, top_k: int = 1) -> list[tuple[str, str, float]]:
    """
    Query the knowledge base index for the top_k most relevant files.

    Returns list of (filename_stem, content_preview, similarity_score),
    ranked best-first. Returns [] if index is empty or unavailable.
    """
    return _query(prompt, KNOWLEDGE_IDX, KNOWLEDGE_DIR, top_k)


def query_memory(prompt: str, top_k: int = 2) -> list[tuple[str, str, float]]:
    """
    Query the session memory index for the top_k most relevant files.

    Returns list of (filename_stem, content_preview, similarity_score),
    ranked best-first. Returns [] if index is empty or unavailable.
    """
    return _query(prompt, MEMORY_IDX, MEMORY_DIR, top_k)


# ─── Targeted Rebuild ─────────────────────────────────────────────────────────

def rebuild_memory() -> int:
    """
    Rebuild only the memory index from AI/memory/.
    Called by close_thread() and write_orchestrator_memory() after writing new
    files so they are immediately queryable. Returns the number of files indexed.
    """
    if not MEMORY_DIR.exists():
        return 0
    return _build_index(MEMORY_DIR, MEMORY_IDX)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def rebuild() -> None:
    """Rebuild both indexes from scratch."""
    if not KNOWLEDGE_DIR.exists():
        print(f"[rag] WARN: knowledge dir not found: {KNOWLEDGE_DIR}")
        n_knowledge = 0
    else:
        print(f"[rag] Indexing knowledge: {KNOWLEDGE_DIR}")
        n_knowledge = _build_index(KNOWLEDGE_DIR, KNOWLEDGE_IDX)
        print(f"[rag] Knowledge index: {n_knowledge} files → {KNOWLEDGE_IDX}")

    if not MEMORY_DIR.exists():
        print(f"[rag] WARN: memory dir not found: {MEMORY_DIR}")
        n_memory = 0
    else:
        print(f"[rag] Indexing memory:    {MEMORY_DIR}")
        n_memory = _build_index(MEMORY_DIR, MEMORY_IDX)
        print(f"[rag] Memory index:    {n_memory} files → {MEMORY_IDX}")

    print(f"\n[rag] Done. knowledge={n_knowledge} files, memory={n_memory} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG index builder for local AI pipeline")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild both indexes from scratch")
    args = parser.parse_args()

    if args.rebuild:
        rebuild()
    else:
        parser.print_help()
