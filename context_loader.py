#!/usr/bin/env python3
# context_loader.py — Dynamic ground-truth context injector for pipeline pipeline
# v5 — Priority Zero compliance: remove subprocess pip install
#
# Changes from v4:
#   - SECURITY (Priority Zero rule #1): Removed subprocess.check_call pip install
#     of rank_bm25 that fired on first import when the package was missing. That
#     was an external network call on the pipeline path without explicit approval.
#     rank_bm25 must now be installed manually before running the pipeline
#     (see requirements.txt and README.md). The _BM25_AVAILABLE = False fallback
#     path is unchanged — cosine-only mode continues to work without rank_bm25.
#
# Changes from v3:
#   - get_file_manifest() output capped at 200 chars: shows first 10 files then
#     "... and N more". File discovery logic unchanged — only the rendered output
#     is truncated. Frees ~300 chars on every query for retrieval content.
#   - CONTEXT_INJECT_LIMIT raised 2000 → 4000 in config.py (not here — the
#     constant lives there). Memory section was being cut off by the old limit
#     when the manifest + knowledge sections filled the budget first.
#
# Changes from v2:
#   - BM25 (rank_bm25) added as a parallel retrieval path alongside cosine in
#     search_knowledge_base() and load_recent_memory(). Both paths run on every
#     query; results are merged and deduplicated by (source_file, chunk_text).
#   - BM25 indexes .md files at paragraph level (double-newline split), not file
#     level. Finer granularity gives BM25 better TF signal — a paragraph containing
#     "nomic-embed-text" twice scores higher than a 1500-char file where the term
#     appears once. Cosine remains file-level (unchanged).
#   - Tokenizer preserves hyphenated tokens intact: "nomic-embed-text",
#     "mxbai-embed-large", version strings like "v3.3", tool names. Splits on
#     whitespace and punctuation except hyphens. Technical exact-match queries
#     that cosine misses consistently hit on BM25.
#   - BM25 index is NOT persisted to disk — rebuilt from source .md files at module
#     import time in milliseconds. No new pkl files. No disk writes for BM25.
#   - rank_bm25 must be installed manually (pip install rank_bm25 or via
#     requirements.txt). If not installed, BM25 is disabled and the module falls
#     back to cosine-only silently. Never crashes context_loader regardless of
#     rank_bm25 availability.
#   - Why BM25 was added: cosine similarity on nomic-embed-text embeddings fails for
#     low-frequency technical tokens — model names, version strings, tool names. These
#     tokens are rare in the embedding model's training distribution so their vectors
#     carry weak specific meaning. BM25 is exact token frequency matching — it finds
#     "nomic-embed-text" in a document because the string is literally there. The two
#     methods are complementary: cosine for conceptual relevance, BM25 for exact terms.
#
# Changes from v1:
#   - search_knowledge_base() now uses rag_index.query_knowledge() (cosine similarity)
#     instead of keyword scoring. Threshold: similarity > 0.25.
#   - load_recent_memory() now uses rag_index.query_memory() (relevance-ranked)
#     instead of most-recently-modified-file ordering.
#   - Keyword helper functions (_STOPWORDS, _prompt_words, _keyword_score) removed.
#   - Added import of rag_index module.
#
# WHAT THIS IS:
#   Skills teach the model how to behave.
#   context-loader tells the model what is real — what files exist,
#   what has been built, what the knowledge base contains, what happened recently.
#
# WHAT THIS IS NOT:
#   Not a replacement for skills or constraints.
#
# INTEGRATION:
#   Called from load_skill() in run_task.py.
#   Output is injected AFTER skill content, BEFORE hard constraints.
#   Constraint order is preserved — nothing from this file can override constraints.
#
# PATHS:
#   Router dir:    <repo>/
#   Knowledge dir: <vault>/AI/knowledge/
#   Memory dir:    <vault>/AI/memory/
#
# LIMIT:
#   CONTEXT_INJECT_LIMIT is defined in config.py — do not reuse SKILL_INJECT_LIMIT
#   or BASE_INJECT_LIMIT. This module has its own budget.

import re
from pathlib import Path
from config import VAULT, CONTEXT_INJECT_LIMIT
import rag_index

# ─── Paths ────────────────────────────────────────────────────────────────────

ROUTER_DIR    = Path(__file__).parent.resolve()
KNOWLEDGE_DIR = VAULT / "AI/knowledge"
MEMORY_DIR    = VAULT / "AI/memory"


# ─── BM25 Support ─────────────────────────────────────────────────────────────

# rank_bm25 is optional — install manually via requirements.txt before running.
# _BM25_AVAILABLE controls whether BM25 paths are attempted.
try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


def _tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25, preserving hyphenated technical tokens intact.

    Splits on whitespace and common punctuation but NOT hyphens. This ensures
    "nomic-embed-text" and "mxbai-embed-large" remain single tokens and match
    documents that contain exactly those strings.

    Lowercases all tokens. Filters empty strings.
    """
    tokens = re.split(r'[\s,;:!()\[\]{}<>/"\'\\@#$%^&*+=~|.]+', text.lower())
    return [t for t in tokens if t]


class _BM25Dir:
    """
    BM25 index over all .md files in a directory, chunked at paragraph level.

    Built at construction time from the source directory — no persistence.
    Each paragraph from each file becomes one BM25 document.
    Rebuilds automatically if the source directory changes (restart required).
    """

    def __init__(self, source_dir: Path) -> None:
        self.docs: list[tuple[str, str]] = []   # (stem, paragraph_text)
        self._bm25 = None

        if not _BM25_AVAILABLE or not source_dir.exists():
            return

        corpus: list[list[str]] = []
        for f in sorted(source_dir.glob("*.md")):
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            stem = f.stem
            # Paragraph-level chunking — double newline as separator
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for para in paragraphs:
                self.docs.append((stem, para))
                corpus.append(_tokenize(para))

        if corpus:
            try:
                self._bm25 = _BM25Okapi(corpus)
            except Exception:
                self._bm25 = None

    def search(self, query: str, top_k: int) -> list[tuple[str, str, float]]:
        """
        Return top_k (stem, paragraph_text, bm25_score) results for query.
        Only returns results with score > 0 — zero means no token overlap.
        """
        if self._bm25 is None or not self.docs:
            return []

        tokens = _tokenize(query)
        try:
            scores = self._bm25.get_scores(tokens)
        except Exception:
            return []

        top_k = min(top_k, len(scores))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            (self.docs[i][0], self.docs[i][1], float(scores[i]))
            for i in ranked
            if scores[i] > 0
        ]


# Build BM25 indexes at import time — fast, no Ollama call, no disk write
_kb_bm25  = _BM25Dir(KNOWLEDGE_DIR)
_mem_bm25 = _BM25Dir(MEMORY_DIR)


def _merge(
    cosine: list[tuple[str, str, float]],
    bm25:   list[tuple[str, str, float]],
    max_results: int = 4,
) -> list[tuple[str, str, float]]:
    """
    Merge cosine and BM25 results, deduplicate by (stem, chunk_text[:100]).
    Cosine results are listed first (semantic relevance takes priority for display
    ordering). BM25 results are appended after, skipping exact-text duplicates.
    Returns up to max_results entries.
    """
    seen: set[tuple[str, str]] = set()
    merged: list[tuple[str, str, float]] = []

    for stem, text, score in cosine + bm25:
        key = (stem, text[:100])
        if key not in seen:
            seen.add(key)
            merged.append((stem, text, score))
        if len(merged) >= max_results:
            break

    return merged


# ─── Section 1: File Manifest ─────────────────────────────────────────────────

def get_file_manifest() -> str:
    """
    List all files in the router directory with sizes.
    Gives the model awareness of what scripts exist — prevents hallucinating
    filenames or claiming files don't exist.

    Returns empty string if directory is missing or empty.
    """
    if not ROUTER_DIR.exists():
        return ""

    files = sorted(f for f in ROUTER_DIR.iterdir() if f.is_file())
    if not files:
        return ""

    lines = []
    for f in files:
        size_kb = f.stat().st_size / 1024
        lines.append(f"  {f.name} ({size_kb:.1f}KB)")

    _MAX_MANIFEST_CHARS = 200
    header = "## Router Directory — Current Files\n"
    output = header
    shown = 0
    for line in lines:
        candidate = output + line + "\n"
        if len(candidate) > _MAX_MANIFEST_CHARS:
            remaining = len(lines) - shown
            if remaining:
                output += f"  ... and {remaining} more"
            break
        output = candidate
        shown += 1

    return output


# ─── Section 2: Relevant File Injection ───────────────────────────────────────

def inject_referenced_files(prompt: str) -> str:
    """
    If the prompt names a file that exists in the router directory, inject
    its content so the model can see exactly what's in it.

    Matching logic:
      - Checks full filename (e.g., "router.py")
      - Checks stem with underscores and hyphens (e.g., "run_task", "run-task")
      - Case-insensitive

    Only the first match is injected (one file = one block).
    Content is capped at 800 chars — this is bonus context, not a skill.
    Returns empty string if no match.
    """
    if not ROUTER_DIR.exists():
        return ""

    router_files = {f.name: f for f in ROUTER_DIR.iterdir() if f.is_file()}
    prompt_lower = prompt.lower()

    for filename, path in sorted(router_files.items()):
        stem = path.stem.lower()
        stem_alt = stem.replace("_", "-")  # run_task → run-task

        if (
            filename.lower() in prompt_lower
            or stem in prompt_lower
            or stem_alt in prompt_lower
        ):
            try:
                content = path.read_text()[:800]
                return (
                    f"## Referenced File: {filename}\n"
                    f"```python\n{content}\n```"
                )
            except Exception:
                return ""

    return ""


# ─── Section 3: Knowledge Base Search ────────────────────────────────────────

_KB_SIMILARITY_THRESHOLD = 0.25


def search_knowledge_base(prompt: str) -> str:
    """
    Hybrid search over AI/knowledge/ — cosine similarity + BM25.

    Cosine path: delegates to rag_index.query_knowledge(). File-level embeddings
    via nomic-embed-text. Only includes results above _KB_SIMILARITY_THRESHOLD.

    BM25 path: paragraph-level index rebuilt at import time. Exact token matching
    — finds "nomic-embed-text", version strings, tool names that cosine misses.
    No threshold — any positive BM25 score is included.

    Results are merged and deduplicated by (stem, chunk_text[:100]). Cosine
    results appear first. Combined output is labelled with its source file.

    Returns empty string if neither path finds anything.
    """
    # Cosine (file-level)
    cosine_hits = [
        (stem, content, score)
        for stem, content, score in rag_index.query_knowledge(prompt, top_k=2)
        if score >= _KB_SIMILARITY_THRESHOLD
    ]

    # BM25 (paragraph-level) — no-op if rank_bm25 unavailable
    bm25_hits = _kb_bm25.search(prompt, top_k=2)

    merged = _merge(cosine_hits, bm25_hits, max_results=3)
    if not merged:
        return ""

    parts = []
    for stem, text, _score in merged:
        parts.append(f"### {stem}\n{text[:600]}")

    return "## Knowledge Base\n" + "\n\n".join(parts)


# ─── Section 4: Recent Memory ─────────────────────────────────────────────────

def load_recent_memory(prompt: str = "", n: int = 2) -> str:
    """
    Hybrid search over AI/memory/ — cosine similarity + BM25.

    Cosine path: rag_index.query_memory(), relevance-ranked, top n results.
    No threshold applied — memory files are generally relevant if they score at all.

    BM25 path: paragraph-level index rebuilt at import time. Finds exact technical
    terms in session memory that cosine would miss.

    Results from both paths are merged and deduplicated. Total capped at n+2
    entries before formatting.

    Returns empty string if no .md memory files exist yet.
    """
    # Cosine (file-level)
    cosine_hits = [
        (stem, content, score)
        for stem, content, score in rag_index.query_memory(prompt, top_k=n)
    ]

    # BM25 (paragraph-level)
    bm25_hits = _mem_bm25.search(prompt, top_k=n)

    merged = _merge(cosine_hits, bm25_hits, max_results=n + 2)
    if not merged:
        return ""

    parts = []
    for stem, text, _score in merged:
        parts.append(f"### {stem}\n{text[:600]}")

    return "## Recent Session Memory\n" + "\n\n".join(parts)


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def load_context(prompt: str, task: str = "") -> str:
    """
    Assemble the dynamic context block for injection into the system prompt.

    Called from load_skill() in run_task.py. Output slot:
      [base context] → [skill content] → [THIS BLOCK] → [hard constraints]

    Assembles four sections in order:
      1. File manifest    — what router scripts exist on disk
      2. Referenced file  — content of a file named in the prompt (if any)
      3. Knowledge chunk  — hybrid cosine+BM25 search over knowledge base
      4. Recent memory    — hybrid cosine+BM25 search over session memory

    Output is capped to CONTEXT_INJECT_LIMIT chars (defined in config.py).
    Returns empty string if nothing useful was found — never crashes.

    Args:
        prompt: The raw user prompt (used for search).
        task:   The dominant task string from routing (reserved for future use).
    """
    sections = []

    # 1. File manifest — always attempt; short and always informative
    try:
        manifest = get_file_manifest()
        if manifest:
            sections.append(manifest)
    except Exception as e:
        print(f"[context-loader] WARN: file manifest failed: {e}")

    # 2. Referenced file — only fires if a filename is detected in the prompt
    try:
        ref_file = inject_referenced_files(prompt)
        if ref_file:
            sections.append(ref_file)
    except Exception as e:
        print(f"[context-loader] WARN: file injection failed: {e}")

    # 3. Knowledge base — hybrid cosine+BM25
    try:
        kb_chunk = search_knowledge_base(prompt)
        if kb_chunk:
            sections.append(kb_chunk)
    except Exception as e:
        print(f"[context-loader] WARN: knowledge search failed: {e}")

    # 4. Recent memory — hybrid cosine+BM25
    try:
        memory = load_recent_memory(prompt=prompt, n=2)
        if memory:
            sections.append(memory)
    except Exception as e:
        print(f"[context-loader] WARN: memory load failed: {e}")

    if not sections:
        return ""

    block = "\n\n".join(sections)
    return block[:CONTEXT_INJECT_LIMIT]


# ─── CLI (diagnostic only) ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    prompt = " ".join(sys.argv[1:]) or "test prompt"
    print(f"[context-loader] Testing with prompt: {prompt!r}")
    print(f"[context-loader] BM25 available: {_BM25_AVAILABLE}")
    if _BM25_AVAILABLE:
        print(f"[context-loader] BM25 knowledge docs: {len(_kb_bm25.docs)}")
        print(f"[context-loader] BM25 memory docs:    {len(_mem_bm25.docs)}")
    print()
    print("─" * 60)

    manifest = get_file_manifest()
    print(f"[manifest]         {len(manifest)} chars" if manifest else "[manifest]         nothing found")

    ref = inject_referenced_files(prompt)
    print(f"[ref-file]         {len(ref)} chars" if ref else "[ref-file]         no match")

    # Knowledge — cosine
    kb_cosine = rag_index.query_knowledge(prompt, top_k=2)
    if kb_cosine:
        for stem, _content, score in kb_cosine:
            hit = "HIT" if score >= _KB_SIMILARITY_THRESHOLD else f"BELOW threshold ({_KB_SIMILARITY_THRESHOLD})"
            print(f"[knowledge/cosine] {stem!r}  score={score:.4f}  [{hit}]")
    else:
        print("[knowledge/cosine] no match")

    # Knowledge — BM25
    if _BM25_AVAILABLE:
        kb_bm25_results = _kb_bm25.search(prompt, top_k=2)
        if kb_bm25_results:
            for stem, para, score in kb_bm25_results:
                print(f"[knowledge/bm25]   {stem!r}  score={score:.2f}  preview={para[:80]!r}")
        else:
            print("[knowledge/bm25]   no match")
    else:
        print("[knowledge/bm25]   rank_bm25 not available")

    # Memory — cosine
    mem_cosine = rag_index.query_memory(prompt, top_k=2)
    if mem_cosine:
        for stem, _content, score in mem_cosine:
            print(f"[memory/cosine]    {stem!r}  score={score:.4f}")
    else:
        print("[memory/cosine]    no files found")

    # Memory — BM25
    if _BM25_AVAILABLE:
        mem_bm25_results = _mem_bm25.search(prompt, top_k=2)
        if mem_bm25_results:
            for stem, para, score in mem_bm25_results:
                print(f"[memory/bm25]      {stem!r}  score={score:.2f}  preview={para[:80]!r}")
        else:
            print("[memory/bm25]      no match")
    else:
        print("[memory/bm25]      rank_bm25 not available")

    print("─" * 60)
    result = load_context(prompt)
    print(f"\n[load_context output — {len(result)} chars / {CONTEXT_INJECT_LIMIT} limit]\n")
    print(result)
