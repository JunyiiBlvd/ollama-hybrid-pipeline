---
title: Example Knowledge Document
tags: [example, architecture, fastapi]
created: 2026-01-01
---

# Example Knowledge Document

This file lives in `vault/AI/knowledge/` and is indexed by `rag_index.py` for
hybrid BM25+cosine retrieval. When a user prompt is relevant to this document,
its content will be injected into the model's context automatically.

## What Belongs Here

Knowledge documents capture stable facts about your project that the model
should know — architecture decisions, API behavior, model specs, resolved bugs.
They are NOT session logs or task prompts.

Good candidates:
- "How the authentication middleware works"
- "Why we switched from polling to websockets"
- "The correct way to instantiate ModelLoader"
- "Benchmark results: BM25 score=17.74 vs cosine miss on 'nomic-embed-text'"

## Format Guidelines

- Use clear headers so BM25 can find sections by keyword
- Include exact terms the model needs to recognize (model names, class names, flags)
- Keep files focused — one topic per file retrieves better than mixed-topic files
- First paragraph should state the core fact clearly (it becomes the preview in RAG results)

## Example: Architecture Decision

**Decision:** Use hybrid BM25+cosine retrieval instead of cosine-only.

**Context:** Cosine similarity on nomic-embed-text embeddings fails for
low-frequency technical tokens. A query for "nomic-embed-text" returned
the wrong document (score=0.31) while BM25 found the correct one with
score=17.74 — exact token frequency matching beats semantic similarity
for exact technical terms.

**Result:** context_loader.py runs both paths in parallel and merges results,
cosine first (semantic priority), BM25 appended for exact-term hits.
