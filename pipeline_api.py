#!/usr/bin/env python3
# pipeline_api.py — OpenAI-compatible HTTP wrapper for the pipeline pipeline
# v10 — Wired call_ollama_safe() to backend.py adapter
#
# Changes from v9:
# - Replaced direct Ollama POST in call_ollama_safe() with call_backend() from backend.py
# - No interface changes — callers unaffected
# - Known gap: _stream_pipeline() still calls Ollama directly (not wired in this pass)
#
# Changes from v8 (now v9):
#   - SECURITY: uvicorn.run host changed 0.0.0.0 → 127.0.0.1. The 0.0.0.0
#     binding was the FastAPI default, not a deliberate LAN-trust decision.
#     With write_vault_file and orchestrator live, LAN exposure is no longer
#     bounded. Open WebUI uses --network=host so it reaches 127.0.0.1:11436
#     unchanged. No auth layer added — loopback is sufficient for single-user.
#
# Changes from v7:
#   - log_interaction() gains eval_passed (bool|null), eval_score (int|null),
#     and eval_failures (list[{type,severity,issue}]|null) parameters.
#   - sync path: first-attempt log stores first-attempt eval; retried output
#     is re-evaluated before final log so eval fields match actual output.
#   - streaming path writes null eval fields (eval requires full response).
#
# v7 — Stop sequence for multi-turn fake user turns
#
# Changes from v6:
#   - Ollama API calls now include stop=["User:", "\nUser:"] to prevent
#     model from generating fake user turns in multi-turn sessions
#
# v6 — Footer sentinel collision fix
#
# Changes from v5:
#   - Footer sentinel changed from "\n\n---\n" to "\n\n<<<PIPELINE_FOOTER>>>\n"
#     — prevents collision with markdown horizontal rules in model output
#
# v5 — Resilient SSE error handling
#
# Changes from v4:
#   - _stream_pipeline(): except Exception added after ConnectionError/Timeout catches;
#     all unhandled exceptions now yield a clean SSE error chunk instead of crashing
#     the generator and causing ChunkedEncodingError on the client
#   - _stream_orchestrator(): stderr logging added to existing exception handlers;
#     error chunk updated to include exception type
#   - exception type, message, and model logged to stderr on stream failure in both
#     streaming generators
#
# v4 — Orchestrator model + streaming log
#
# Changes from v3:
#   - local-orchestrator added as a second model. Requests with
#     model=local-orchestrator are routed to run_orchestrator() instead of
#     the regular pipeline. For stream=True, yields an immediate status chunk
#     then the full orchestrator output. For stream=False, runs synchronously
#     and returns the final step output. Exposes agentic multi-step goal
#     execution from Open WebUI — select local-orchestrator, send a goal.
#   - Streaming path now logs to routing-log.jsonl. Previously all stream=True
#     requests were invisible to the evaluator and routing accuracy system.
#     Tokens are buffered in _stream_pipeline and logged after [DONE] with
#     source="api-stream". Streaming requests now appear in log_view.py output.
#
# Changes from v2:
#   - stream=True requests now return a StreamingResponse using SSE (Server-Sent
#     Events) in OpenAI chunk format. Ollama is called with stream=True and each
#     token chunk is forwarded immediately, giving Open WebUI live token output
#     instead of a frozen wait followed by a full response dump.
#   - Streaming path skips evaluation and retry (eval requires the full response).
#     Evaluation and retry remain active on the non-streaming path.
#   - _stream_pipeline() added — runs routing/skill/context synchronously, then
#     streams Ollama tokens. Appends routing metadata as the final chunk.
#   - StreamingResponse imported from fastapi.responses.
#
# Changes from v1:
#   - route() now returns a 4-tuple (selected_model, task_scores, model_scores,
#     router_path) since router.py v3.3. Fixed unpack at line ~241 that was
#     only capturing 3 values (would crash on any API request).
#   - router_path ("keyword", "llm", "default") added to log entries.
#
# Changes from v1 (original):
#   - Exposes GET /v1/models and POST /v1/chat/completions
#   - Full pipeline runs on each request: route → skill → context → constraints
#     → Ollama → evaluate → retry once on critical failure → log
#   - call_ollama_safe() replaces run_task.call_ollama() — raises HTTPException
#     instead of sys.exit() so the server stays alive on Ollama errors
#   - Extracts last user message from OpenAI messages array
#   - Multi-turn context: all prior messages passed as conversation history
#     in the prompt (Ollama /api/generate has no native multi-turn support)
#   - Pipeline metadata injected as a comment in the response body so
#     Open WebUI users can see which skill/model was selected
#   - Runs on 127.0.0.1:11436 — reachable from Open WebUI's --network host
#
# Usage:
#   pip install fastapi uvicorn
#   cd <repo>/
#   python pipeline_api.py
#
# Open WebUI setup:
#   Settings → Connections → OpenAI-Compatible APIs
#   URL:     http://127.0.0.1:11436
#   API Key: pipeline (any non-empty string)
#   Model will appear as: local-pipeline

import os
import sys
import json
import time
import uuid
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# Add router directory to path so pipeline imports work regardless of cwd
ROUTER_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROUTER_DIR))

from router import route, check_disambiguation
from config import (
    OLLAMA_URL, SKILL_FILES, BASE_CONTEXT_FILE,
    SKILL_INJECT_LIMIT, BASE_INJECT_LIMIT, LOG_FILE, TASK_PRIORITY
)
from constraints import load_constraints
from context_loader import load_context
from evaluator import evaluate
from orchestrator import run_orchestrator
from backend import call_backend, BackendConnectionError

# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="pipeline pipeline API", version="1.0.0")

PORT = int(os.getenv("PIPELINE_PORT", "11436"))
MODEL_ID = "local-pipeline"
ORCHESTRATOR_MODEL_ID = "local-orchestrator"


# ─── OpenAI Request / Response Models ────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[Message]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


# ─── Pipeline Helpers ─────────────────────────────────────────────────────────

def load_skill(task_scores: dict, prompt: str = "") -> tuple[str, str]:
    """
    Identical logic to run_task.load_skill().
    Duplicated here so pipeline_api has no import dependency on run_task.
    """
    system_parts = []
    skill_loaded = "base-only"

    if BASE_CONTEXT_FILE.exists():
        base_content = BASE_CONTEXT_FILE.read_text()[:BASE_INJECT_LIMIT]
        system_parts.append(base_content)

    dominant_task = None
    dominant_score = 0

    for task in TASK_PRIORITY:
        score = task_scores.get(task, 0)
        if score > dominant_score and task in SKILL_FILES:
            if SKILL_FILES[task].exists():
                dominant_task = task
                dominant_score = score

    if dominant_task:
        skill_path = SKILL_FILES[dominant_task]
        if skill_path.exists():
            skill_content = skill_path.read_text()[:SKILL_INJECT_LIMIT]
            system_parts.append(skill_content)
            skill_loaded = dominant_task

            context_block = load_context(prompt, dominant_task)
            if context_block:
                system_parts.append(context_block)

            constraint_block = load_constraints(dominant_task)
            if constraint_block:
                system_parts.append(constraint_block)
        else:
            # Fallback to next available skill
            for task, score in sorted(task_scores.items(), key=lambda x: -x[1]):
                if task == dominant_task:
                    continue
                if task in SKILL_FILES and SKILL_FILES[task].exists():
                    skill_content = SKILL_FILES[task].read_text()[:SKILL_INJECT_LIMIT]
                    system_parts.append(skill_content)
                    skill_loaded = f"{task}(fallback-from-{dominant_task})"
                    break

    return "\n\n---\n\n".join(system_parts), skill_loaded


def build_prompt_with_disambiguation(prompt: str) -> str:
    """Prepend disambiguation clarifications if any ambiguous terms detected."""
    clarifications = check_disambiguation(prompt)
    if not clarifications:
        return prompt
    clarification_block = "\n".join(f"[CONTEXT] {c}" for c in clarifications)
    return f"{clarification_block}\n\n{prompt}"


def build_multi_turn_prompt(messages: list[Message], final_prompt: str) -> str:
    """
    Flatten conversation history into a single prompt string.
    Ollama /api/generate has no native multi-turn support —
    prior turns are prepended as labelled blocks.
    Only includes turns before the final user message.
    """
    history = messages[:-1]  # all but the last user message
    if not history:
        return final_prompt

    turns = []
    for msg in history:
        label = "User" if msg.role == "user" else "Assistant"
        turns.append(f"{label}: {msg.content}")

    history_block = "\n".join(turns)
    return f"[Conversation history]\n{history_block}\n\n[Current request]\n{final_prompt}"


def call_ollama_safe(model: str, prompt: str, system: str = "") -> str:
    """
    Call inference backend. Raises HTTPException instead of sys.exit() —
    keeps the server alive on backend errors.
    """
    try:
        return call_backend(
            model,
            prompt,
            stream=False,
            temperature=0.7,
            system=system,
            stop=["User:", "\nUser:"],
        )
    except BackendConnectionError as e:
        if isinstance(e.original, requests.exceptions.Timeout):
            raise HTTPException(
                status_code=504,
                detail="Ollama timed out after 180 seconds."
            )
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Run: sudo systemctl start ollama"
        )
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama API error: {e}")


def log_interaction(
    prompt: str,
    selected_model: str,
    task_scores: dict,
    model_scores: dict,
    skill_loaded: str,
    output: str,
    disambiguated: bool,
    retried: bool = False,
    source: str = "api",
    router_path: str = "default",
    eval_passed: bool | None = None,
    eval_score: int | None = None,
    eval_failures: list | None = None,
):
    """Write one JSONL entry to routing-log.jsonl. source='api' distinguishes
    web UI requests from CLI runs."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "ts":                 datetime.now().isoformat(),
        "prompt":             prompt[:200],
        "model":              selected_model,
        "router_path":        router_path,
        "task_scores":        {k: v for k, v in task_scores.items() if v > 0},
        "model_scores":       {k: round(v, 2) for k, v in model_scores.items()},
        "skill_loaded":       skill_loaded,
        "disambiguated":      disambiguated,
        "constraint_injected": skill_loaded if load_constraints(skill_loaded) else False,
        "output_len":         len(output),
        "output_preview":     output[:300],
        "correct":            None,
        "notes":              "",
        "retried":            retried,
        "source":             source,
        "eval_passed":        eval_passed,
        "eval_score":         eval_score,
        "eval_failures":      eval_failures,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ─── Pipeline Core ────────────────────────────────────────────────────────────

def run_pipeline(messages: list[Message]) -> tuple[str, dict]:
    """
    Run the full pipeline for a list of messages.
    Returns: (final_output, metadata_dict)
    """
    # Extract last user message as the primary prompt
    user_messages = [m for m in messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message in request")
    prompt = user_messages[-1].content

    # Route
    selected_model, task_scores, model_scores, router_path = route(prompt)

    # Skill + context + constraints
    system_context, skill_loaded = load_skill(task_scores, prompt)

    # Disambiguation
    clarifications = check_disambiguation(prompt)
    disambiguated = len(clarifications) > 0
    prompt_to_send = build_prompt_with_disambiguation(prompt)

    # Multi-turn history injection
    prompt_to_send = build_multi_turn_prompt(messages, prompt_to_send)

    # Call Ollama (first attempt)
    output = call_ollama_safe(selected_model, prompt_to_send, system=system_context)

    # Evaluate — retry once on critical failure
    retried = False
    eval_result = evaluate(output, skill_loaded)
    if not eval_result.passed:
        critical_issues = [f.issue for f in eval_result.failures if f.severity == "critical"]
        if critical_issues:
            # Log failed first attempt
            log_interaction(
                prompt=prompt,
                selected_model=selected_model,
                task_scores=task_scores,
                model_scores=model_scores,
                skill_loaded=skill_loaded,
                output=output,
                disambiguated=disambiguated,
                retried=False,
                source="api",
                router_path=router_path,
                eval_passed=eval_result.passed,
                eval_score=round(eval_result.score * 100),
                eval_failures=[{"type": f.type, "severity": f.severity, "issue": f.issue} for f in eval_result.failures],
            )
            # Build retry prompt
            issues_text = "\n".join(f"- {issue}" for issue in critical_issues)
            retry_prompt = (
                f"[EVAL RETRY] Your previous response had these issues:\n"
                f"{issues_text}\n\n"
                f"{prompt_to_send}"
            )
            output = call_ollama_safe(selected_model, retry_prompt, system=system_context)
            retried = True
            eval_result = evaluate(output, skill_loaded)

    # Log final output
    log_interaction(
        prompt=prompt,
        selected_model=selected_model,
        task_scores=task_scores,
        model_scores=model_scores,
        skill_loaded=skill_loaded,
        output=output,
        disambiguated=disambiguated,
        retried=retried,
        source="api",
        router_path=router_path,
        eval_passed=eval_result.passed,
        eval_score=round(eval_result.score * 100),
        eval_failures=[{"type": f.type, "severity": f.severity, "issue": f.issue} for f in eval_result.failures],
    )

    metadata = {
        "model":         selected_model,
        "router_path":   router_path,
        "skill_loaded":  skill_loaded,
        "retried":       retried,
        "disambiguated": disambiguated,
        "task_scores":   {k: v for k, v in task_scores.items() if v > 0},
    }

    return output, metadata


# ─── Streaming Pipeline ───────────────────────────────────────────────────────

def _stream_pipeline(messages: list[Message]):
    """
    Generator for SSE streaming. Runs routing and skill loading synchronously,
    then streams Ollama token chunks in OpenAI chunk format.

    Evaluation and retry are skipped on the streaming path — they require the
    full response to be buffered before scoring, which defeats streaming.
    Routing metadata is appended as the final content chunk before [DONE].
    """
    # Extract prompt
    user_messages = [m for m in messages if m.role == "user"]
    if not user_messages:
        yield "data: [DONE]\n\n"
        return

    prompt = user_messages[-1].content

    # Route + skill + context + constraints (synchronous — happens before first token)
    selected_model, task_scores, model_scores, router_path = route(prompt)
    system_context, skill_loaded = load_skill(task_scores, prompt)
    clarifications = check_disambiguation(prompt)
    prompt_to_send = build_prompt_with_disambiguation(prompt)
    prompt_to_send = build_multi_turn_prompt(messages, prompt_to_send)

    chunk_id  = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created   = int(time.time())

    def _make_chunk(content: str, finish_reason=None) -> str:
        payload = {
            "id":      chunk_id,
            "object":  "chat.completion.chunk",
            "created": created,
            "model":   MODEL_ID,
            "choices": [{
                "index":         0,
                "delta":         {"content": content},
                "finish_reason": finish_reason,
            }],
        }
        return f"data: {json.dumps(payload)}\n\n"

    # Stream from Ollama — buffer tokens for post-stream logging
    payload = {
        "model":   selected_model,
        "prompt":  prompt_to_send,
        "stream":  True,
        "stop":    ["User:", "\nUser:"],
        "options": {"temperature": 0.7},
    }
    if system_context:
        payload["system"] = system_context

    output_tokens: list[str] = []

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=180) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk_data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk_data.get("response", "")
                if token:
                    output_tokens.append(token)
                    yield _make_chunk(token)
                if chunk_data.get("done"):
                    break

    except requests.exceptions.ConnectionError:
        yield _make_chunk("\n\n[ERROR] Cannot connect to Ollama.")
        yield "data: [DONE]\n\n"
        return
    except requests.exceptions.Timeout:
        yield _make_chunk("\n\n[ERROR] Ollama timed out.")
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        print(f"[stream error] {type(e).__name__}: {e} | model: {selected_model}", file=sys.stderr)
        yield _make_chunk(f"\n\n[ERROR] Stream failed: {type(e).__name__}")
        yield "data: [DONE]\n\n"
        return

    # Log the completed streaming interaction — previously a data blackhole
    log_interaction(
        prompt=prompt,
        selected_model=selected_model,
        task_scores=task_scores,
        model_scores=model_scores,
        skill_loaded=skill_loaded,
        output="".join(output_tokens),
        disambiguated=len(clarifications) > 0,
        retried=False,
        source="api-stream",
        router_path=router_path,
        eval_passed=None,
        eval_score=None,
        eval_failures=None,
    )

    # Append routing metadata as final chunk
    meta = (
        f"\n\n<<<PIPELINE_FOOTER>>>\n"
        f"*routed → {selected_model} | skill: {skill_loaded} | path: {router_path}*"
    )
    if clarifications:
        meta += "\n*ℹ disambiguation injected*"
    yield _make_chunk(meta)

    # Done
    yield _make_chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"


# ─── Orchestrator Streaming ───────────────────────────────────────────────────

def _stream_orchestrator(messages: list[Message]):
    """
    Sync generator for orchestrator SSE streaming.

    Yields an immediate status chunk so Open WebUI shows activity, then runs
    the full orchestrator (potentially multi-minute, multi-step) synchronously.
    Safe because FastAPI runs sync route handlers in a threadpool — this call
    does not block the event loop.

    SystemExit is caught explicitly: orchestrator's call_ollama (from run_task)
    calls sys.exit() on Ollama connection failure. Catching it here prevents
    that from killing the pipeline API server process.
    """
    user_messages = [m for m in messages if m.role == "user"]
    if not user_messages:
        yield "data: [DONE]\n\n"
        return

    goal = user_messages[-1].content
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created  = int(time.time())

    def _make_chunk(content: str, finish_reason=None) -> str:
        payload = {
            "id":      chunk_id,
            "object":  "chat.completion.chunk",
            "created": created,
            "model":   ORCHESTRATOR_MODEL_ID,
            "choices": [{
                "index":         0,
                "delta":         {"content": content},
                "finish_reason": finish_reason,
            }],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield _make_chunk("*[orchestrator] Decomposing goal into steps…*\n\n")

    try:
        output = run_orchestrator(goal, write_memory=True)
    except SystemExit:
        print(f"[stream error] SystemExit | model: {ORCHESTRATOR_MODEL_ID}", file=sys.stderr)
        yield _make_chunk("\n\n[ERROR] Orchestrator failed — check Ollama connectivity.")
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        print(f"[stream error] {type(e).__name__}: {e} | model: {ORCHESTRATOR_MODEL_ID}", file=sys.stderr)
        yield _make_chunk(f"\n\n[ERROR] Orchestrator error: {type(e).__name__}")
        yield "data: [DONE]\n\n"
        return

    yield _make_chunk(output)
    yield _make_chunk(
        f"\n\n---\n*[orchestrator] goal complete | memory written*",
        finish_reason="stop",
    )
    yield "data: [DONE]\n\n"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/v1/models")
def list_models():
    """Return model list in OpenAI format. Open WebUI calls this on connection."""
    ts = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id":       MODEL_ID,
                "object":   "model",
                "created":  ts,
                "owned_by": "pipeline",
            },
            {
                "id":       ORCHESTRATOR_MODEL_ID,
                "object":   "model",
                "created":  ts,
                "owned_by": "pipeline",
            },
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    """
    Main endpoint. Routes to orchestrator or regular pipeline based on model.

    model=local-orchestrator:
      Goal is passed to run_orchestrator() for multi-step agentic execution.
      stream=True: yields status chunk then full output.
      stream=False: runs synchronously, returns final step output.

    model=local-pipeline (default):
      If request.stream is True: StreamingResponse (SSE), Ollama tokens in
      real time. Evaluation and retry skipped. Logs to routing-log.jsonl
      after stream completes (source="api-stream").
      If request.stream is False: buffers full response, runs evaluation and
      retry, appends routing metadata, returns JSON response.
    """
    # ── Orchestrator path ─────────────────────────────────────────────────────
    if request.model == ORCHESTRATOR_MODEL_ID:
        if request.stream:
            return StreamingResponse(
                _stream_orchestrator(request.messages),
                media_type="text/event-stream",
            )
        # Non-streaming orchestrator
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message in request")
        goal = user_messages[-1].content
        try:
            output = run_orchestrator(goal, write_memory=True)
        except (SystemExit, Exception) as e:
            raise HTTPException(status_code=500, detail=f"Orchestrator error: {e}")
        return {
            "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object":  "chat.completion",
            "created": int(time.time()),
            "model":   ORCHESTRATOR_MODEL_ID,
            "choices": [{
                "index":         0,
                "message":       {"role": "assistant", "content": output},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # ── Regular pipeline path ─────────────────────────────────────────────────
    if request.stream:
        return StreamingResponse(
            _stream_pipeline(request.messages),
            media_type="text/event-stream",
        )

    output, metadata = run_pipeline(request.messages)

    # Append routing metadata as a visible footer in the response
    meta_lines = [
        f"\n\n---",
        f"*routed → {metadata['model']} | skill: {metadata['skill_loaded']}*",
    ]
    if metadata["retried"]:
        meta_lines.append("*⚠ retry fired (evaluator caught critical failure)*")
    if metadata["disambiguated"]:
        meta_lines.append("*ℹ disambiguation injected*")
    if metadata["task_scores"]:
        scores_str = ", ".join(f"{k}:{v}" for k, v in metadata["task_scores"].items())
        meta_lines.append(f"*task signals: {scores_str}*")

    full_response = output + "\n".join(meta_lines)

    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   MODEL_ID,
        "choices": [
            {
                "index":         0,
                "message":       {"role": "assistant", "content": full_response},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "total_tokens":      0,
        },
    }


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "models":  [MODEL_ID, ORCHESTRATOR_MODEL_ID],
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[pipeline_api] starting on port {PORT}")
    print(f"[pipeline_api] pipeline dir: {ROUTER_DIR}")
    print(f"[pipeline_api] models: {MODEL_ID}, {ORCHESTRATOR_MODEL_ID}")
    print(f"[pipeline_api] health: http://127.0.0.1:{PORT}/health")
    print(f"[pipeline_api] Open WebUI connection URL: http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
