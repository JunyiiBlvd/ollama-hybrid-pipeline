# Local AI Pipeline

Our RAG system couldn't find its own documentation.

A query for `"nomic-embed-text"` returned the wrong document using cosine similarity (score: 0.31).
BM25 retrieved the correct document instantly (score: 17.74).

This repository is a local AI pipeline that fixes that failure using hybrid BM25 + cosine retrieval,
along with routing, constraint enforcement, evaluation, and retry — all running locally on Ollama.

---

## What This Is

A Python pipeline that wraps Ollama to provide structured, context-aware inference. Every prompt
passes through a routing layer that classifies task type, loads relevant skill and knowledge context,
enforces hard constraints, evaluates the output, and retries once on critical failure. Logged to JSONL
for accuracy tracking and session memory generation.

Also exposes an OpenAI-compatible HTTP API (`pipeline_api.py`), making it usable from Open WebUI
as a custom model endpoint.

---

## The Problem This Solves

Cosine similarity fails silently on exact technical terms. When querying for `"nomic-embed-text"`,
cosine returned an unrelated document — it treats the term as a semantic concept with no strong
neighbors in the embedding space. BM25 matched it as a literal string and returned the correct
document with score=17.74.

This matters for any knowledge base containing model names, version strings, flag names, file paths,
or tool names. Cosine retrieval works well for conceptual queries ("how does auth work"). It consistently
fails on technical exact-term queries ("why was nomic-embed-text chosen over mxbai-embed-large").

The hybrid retrieval in `context_loader.py` runs both paths in parallel and merges results. Neither
replaces the other — they cover different query types.

---

## Architecture

End-to-end pipeline flow:

```
prompt
  │
  ▼
router.py
  ├── Keyword scorer (primary): score each task category by keyword hits
  │     max score > 0 → select model, return "keyword" path
  │
  └── LLM classifier (fallback): only on zero keyword signal
        success → synthetic task_scores, return "llm" path
        failure → default model, return "default" path
  │
  ▼
run_task.py — load_skill()
  ├── base-context.md (always injected, truncated to BASE_INJECT_LIMIT)
  ├── skill file for dominant task (truncated to SKILL_INJECT_LIMIT)
  ├── context_loader.py → dynamic context block
  │     ├── file manifest (what scripts exist)
  │     ├── referenced file injection (if prompt names a file)
  │     ├── knowledge base search (BM25 + cosine hybrid)
  │     └── session memory search (BM25 + cosine hybrid)
  └── constraints.py → hard constraint block (injected last, never truncated)
  │
  ▼
Ollama /api/generate (or /api/chat for threads)
  │
  ▼
evaluator.py
  └── check_fn rules per task type → EvalResult (passed, failures, score)
  │
  ├── passed → log → print output
  │
  └── critical failure → log first attempt → build retry prompt with
        specific failure reasons → call Ollama again → log final output
```

---

## Key Design Decisions

**Keyword-first routing with LLM fallback on zero-signal only.**
Why: The keyword scorer is deterministic and runs in microseconds. The LLM
classifier adds 1-3 seconds. Calling the LLM on every prompt doubles latency
for no benefit when the prompt already contains recognizable vocabulary. The LLM
only runs when the keyword scorer returns all zeros — ambiguous natural-language
prompts with no domain signal.

**Constraints inject after skill content, never truncated.**
Why: Skill files can be long (up to SKILL_INJECT_LIMIT chars). If constraints
were injected before the skill or interleaved, they could be pushed out of the
model's effective attention window on long contexts. Injecting constraints last
guarantees they are always visible. The constraint block is kept intentionally
short (rules only, no prose).

**BM25 + cosine hybrid, not replacement.**
Why: Cosine similarity is strong for conceptual similarity — "how does the
authentication work" retrieves auth-related documents even without the word
"auth." BM25 is strong for exact technical terms — model names, version strings,
flag names. The two methods are complementary. Replacing cosine with BM25 would
lose semantic retrieval; replacing BM25 with cosine would lose exact-term
retrieval. Both run on every query.

**Retry fires once with specific failure reason prepended.**
Why: A loop that retries until the model gets it right is a latency and cost
trap. One retry with a specific correction prepended ("eval() used — use
json.loads()") is enough to correct a temperature=0.7 drift. If the model fails
the same constraint twice, the constraint or skill file needs updating, not more
retries.

---

## Requirements

- Ollama running locally with at least one model pulled
- `nomic-embed-text` pulled for RAG: `ollama pull nomic-embed-text`
- Python 3.10+

---

## Setup

```bash
# 1. Copy and edit the environment file
cp .env.example .env
# Edit .env — set PIPELINE_VAULT_PATH to your vault directory,
# set PRIMARY_MODEL and FAST_MODEL to models you have pulled

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your vault directory structure
mkdir -p vault/AI/{skills,knowledge,memory,rag-index}

# 4. Build the RAG index (run after adding documents to vault/AI/knowledge/)
python rag_index.py --rebuild

# 5. Run a test prompt
python run_task.py "explain how the router selects a model"
```

## Running the API

```bash
# Start the OpenAI-compatible API server
python pipeline_api.py

# In Open WebUI: Settings → Connections → OpenAI-Compatible APIs
# URL: http://127.0.0.1:11436   (or your PIPELINE_PORT)
# API Key: any non-empty string
# Models: local-pipeline, local-orchestrator
```

## Skill Files

Skill files live in `$PIPELINE_VAULT_PATH/AI/skills/`. Each file is loaded when the router
classifies a prompt to that task type. See `examples/skills/example-skill.md`
for the format.

## Knowledge Base

Knowledge documents live in `$PIPELINE_VAULT_PATH/AI/knowledge/`. Run `python rag_index.py --rebuild`
after adding new documents. See `examples/knowledge/example-knowledge.md` for
the format.
