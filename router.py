# router.py
# v3.3 — route() returns 4-tuple including router_path ("keyword", "llm", "default")
# Changes from v3.2: route() return type extended to (selected_model, task_scores, model_scores, router_path)
#   - "keyword" returned when keyword scorer finds signal (max score > 0)
#   - "llm"     returned when LLM classifier is used (zero keyword signal, LLM succeeds)
#   - "default" returned when both keyword and LLM fail (hard default model used)
#
# Changes from v3.1: route() logic inverted — keyword first, LLM on zero-signal only

import os
import re
import requests
from config import (
    TASK_SIGNALS, MODEL_CAPABILITIES, MODEL_MAP,
    DISAMBIGUATION_MAP, OLLAMA_URL, TASK_PRIORITY,
)

NEGATION_WORDS = ["don't", "dont", "not", "without", "avoid", "no", "never"]

# Classification model — fast, local
_CLASSIFIER_MODEL = os.getenv("FAST_MODEL", "llama3.2")

# Prompt sent to the classifier. Short and directive — no prose, no examples,
# just task names with one-line definitions so the model has enough signal.
_CLASSIFIER_SYSTEM = """\
You are a task classifier for a local AI pipeline.
Given a user prompt, respond with EXACTLY ONE word — the most appropriate task category.
Valid categories (pick one, return only the word):

domain_specialist — queries specific to your configured domain (see your skill file)
linux_admin       — system administration: Docker, containers, systemd, ports, services,
                    logs, firewall, packages, nginx, ssh, mount, permissions
ai_dev            — local AI stack: Ollama, Open WebUI, VRAM, GPU, models, inference,
                    embeddings, Modelfile, context window, temperature
coding            — writing, debugging, or refactoring code (not ai_dev or domain_specialist specific)
session_memory    — generating session memory files or summaries from pipeline log entries:
                    routing-log.jsonl, session summary, daily recap, memory file generation,
                    what was done today, summarize the session
reasoning         — analysis, explanation, comparison, architecture decisions, tradeoffs
fast              — quick lookups, definitions, yes/no questions, simple single-fact answers

Rules:
- Classify by INTENT, not keyword presence.
- If the prompt discusses memory usage of a container, that is linux_admin, not coding.
- If the prompt asks why something works or compares approaches, that is reasoning.
- If the prompt asks to generate, write, or summarize a session memory file: return session_memory.
- Return ONLY the category word. No punctuation, no explanation, nothing else.
"""


# ─── LLM Classifier ───────────────────────────────────────────────────────────

def _classify_with_llm(prompt: str) -> str | None:
    """
    Ask the fast model to classify the prompt into one of the valid task categories.

    Returns:
        A task name from TASK_PRIORITY if classification succeeds.
        None on any error (timeout, unreachable, unrecognized response).

    Constraints:
        - Hard 5-second timeout — this is a fast classifier, not a reasoning call.
        - temperature=0.0 — deterministic output.
        - num_predict=20 — we only need one word; cap tokens to keep it fast.
        - All exceptions are caught — caller never sees an exception from here.
    """
    try:
        payload = {
            "model":  _CLASSIFIER_MODEL,
            "prompt": prompt,
            "system": _CLASSIFIER_SYSTEM,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 20,
            },
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=5)
        response.raise_for_status()

        raw = response.json().get("response", "").strip().lower()

        # Strip any accidental punctuation (period, colon, newline, etc.)
        task = raw.split()[0].strip(".:,\n\r") if raw else ""

        if task in TASK_PRIORITY:
            return task

        # Response was not a recognized task — fall through to keyword scorer
        return None

    except Exception:
        # Timeout, connection error, JSON parse failure, anything — silent fallback
        return None


# ─── Keyword Scorer (fallback) ────────────────────────────────────────────────

def _is_negated(prompt_lower: str, keyword: str) -> bool:
    pattern = r"(?:" + "|".join(NEGATION_WORDS) + r")\s+\w*\s*" + re.escape(keyword)
    return bool(re.search(pattern, prompt_lower))


def score_prompt(prompt: str) -> dict:
    scores = {category: 0 for category in TASK_SIGNALS}
    prompt_lower = prompt.lower()

    for category, keywords in TASK_SIGNALS.items():
        for keyword in keywords:
            if keyword in prompt_lower:
                if _is_negated(prompt_lower, keyword):
                    scores[category] -= 1
                else:
                    scores[category] += 1

    return {k: max(0, v) for k, v in scores.items()}


def score_models(task_scores: dict) -> dict:
    model_scores = {}
    active_tasks = {k: v for k, v in task_scores.items() if v > 0}

    for model, capabilities in MODEL_CAPABILITIES.items():
        if not active_tasks:
            model_scores[model] = 0
            continue
        total = sum(
            task_scores[task] * capabilities.get(task, 0)
            for task in active_tasks
        )
        model_scores[model] = total / len(active_tasks)

    return model_scores


# ─── Disambiguation ───────────────────────────────────────────────────────────

def check_disambiguation(prompt: str) -> list[str]:
    """
    Check if prompt contains ambiguous terms.
    Returns list of clarification strings to prepend to system prompt.
    Only fires if the ambiguous term is NOT accompanied by
    its known context clues (meaning it's probably the generic sense).
    """
    clarifications = []
    prompt_lower = prompt.lower()

    for term, config in DISAMBIGUATION_MAP.items():
        if term in prompt_lower:
            clues_present = any(
                clue in prompt_lower
                for clue in config["context_clues"]
            )
            if not clues_present:
                clarifications.append(config["clarification"])

    return clarifications


# ─── Route ────────────────────────────────────────────────────────────────────

def route(prompt: str) -> tuple[str, dict, dict, str]:
    """
    Classify prompt and select model.

    Classification order:
      1. Keyword scorer (primary) — runs always, fast, deterministic
      2. LLM classifier (zero-signal fallback) — only called when
         keyword scorer returns all zeros (no signal in the prompt)

    Keyword scorer handles all prompts with recognizable vocabulary —
    explicit domain terms, task-specific keywords, compound signals.
    LLM handles the gap: prompts with no keyword hits where intent must
    be inferred from meaning rather than token presence.

    LLM path constructs synthetic task_scores = {task: 1} to keep
    the log schema consistent with keyword path.

    Returns: (selected_model, task_scores, model_scores, router_path)
      router_path is one of: "keyword", "llm", "default"
    """
    # ── Primary: keyword scorer ───────────────────────────────────────────────
    task_scores = score_prompt(prompt)

    if max(task_scores.values()) > 0:
        # Keyword scorer found signal — use it, skip LLM entirely
        model_scores = score_models(task_scores)
        best_model = max(model_scores, key=model_scores.get)
        return best_model, task_scores, model_scores, "keyword"

    # ── Zero-signal fallback: LLM classification ──────────────────────────────
    # Only reached when prompt has no keyword hits at all.
    # LLM infers intent from meaning — handles ambiguous/natural language prompts.
    llm_task = _classify_with_llm(prompt)

    if llm_task is not None:
        # Synthetic scores dict — single winning task at score 1
        task_scores  = {llm_task: 1}
        model_scores = score_models(task_scores)
        best_model   = max(model_scores, key=model_scores.get)
        return best_model, task_scores, model_scores, "llm"

    # ── Hard default: LLM unreachable or returned garbage ────────────────────
    return MODEL_MAP["default"], task_scores, model_scores, "default"
