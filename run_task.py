#!/usr/bin/env python3
# run_task.py — CLI entry point for the pipeline AI pipeline
# v7 — Wired call_ollama() and call_ollama_chat() to backend.py adapter
#
# Changes from v6:
# - Replaced direct Ollama POST in call_ollama() with call_backend()
# - Replaced /api/chat call in call_ollama_chat() with call_backend() + prompt flattening
#   (mirrors build_multi_turn_prompt() logic from pipeline_api.py)
# - No interface changes — callers unaffected
#
# Changes from v5 (now v6):
#   - log_interaction() gains eval_passed (bool|null), eval_score (int|null),
#     and eval_failures (list[{type,severity,issue}]|null) parameters.
#   - First-attempt log on retry path stores first-attempt eval outcome.
#   - Retried output is re-evaluated before final log so eval fields always
#     match the output actually written to the log.
#   - Override path (--model flag) writes null eval fields.
#
# Changes from v4:
#   - --close-thread THREAD_ID flag added: calls close_thread() from
#     thread_store, prints the summary path written, and exits. Does not
#     invoke the pipeline or log a routing entry — only the summarize call
#     inside close_thread() touches Ollama.
#   - close_thread imported from thread_store.
#
# Changes from v3:
#   - route() now returns a 4-tuple; unpacked as (selected_model, task_scores,
#     model_scores, router_path) at the call site (~line 276)
#   - router_path passed to log_interaction() and written to log entry as
#     "router_path" field ("keyword", "llm", or "default")
#   - --show-system flag added: prints the fully assembled system prompt to
#     stdout and exits before making any Ollama API call; lets the user inspect
#     base context + skill + context_loader block + constraints
#
# Changes from v2:
#   - from evaluator import evaluate added
#   - evaluate(output, skill_loaded) called after Ollama on every non-override run
#   - retry fires once on critical failure: logs first attempt (retried=false),
#     prepends specific failure reasons to prompt, calls Ollama a second time
#   - print(output) moved to after evaluate/retry block — only final output ever
#     printed. Previously output was printed before evaluation, meaning both the
#     failed first attempt and retry output appeared in stdout, causing false
#     failures in test_suite when evaluating concatenated output
#   - retried boolean field added to log entry
#
# Changes from v1:
#   - load_skill() added — base context + dominant skill injection via
#     TASK_PRIORITY tie-break, skill fallback on missing file
#   - load_context() integrated — file manifest, referenced files, KB, recent
#     memory injected between skill and constraints (context_loader.py)
#   - load_constraints() integrated — hard rules injected after skill,
#     guaranteed not truncated (constraints.py)
#   - build_prompt_with_disambiguation() added — clarifications prepended when
#     ambiguous terms detected (DISAMBIGUATION_MAP in config.py)
#   - log_interaction() added — full JSONL entry written to routing-log.jsonl
#     after every run including task_scores, model_scores, skill_loaded,
#     disambiguated, constraint_injected, output_len, output_preview, retried
#   - --debug, --no-log, --list-models flags added
#
# Usage:
#   python run_task.py "your prompt here"
#   python run_task.py "your prompt here" --debug
#   python run_task.py "your prompt here" --model the primary model
#   python run_task.py "your prompt here" --no-log
#   python run_task.py --list-models

import sys
import json
import argparse
import requests
from datetime import datetime
from pathlib import Path
from router import route, check_disambiguation
from config import (
    SKILL_FILES, BASE_CONTEXT_FILE,
    SKILL_INJECT_LIMIT, BASE_INJECT_LIMIT, LOG_FILE, TASK_PRIORITY
)
from backend import call_backend, BackendConnectionError
from thread_store import (
    new_thread, load_thread, save_thread, append_exchange,
    get_thread_as_messages, print_threads, close_thread,
)
from constraints import load_constraints
from context_loader import load_context
from evaluator import evaluate

# ─── Context Loading ──────────────────────────────────────────────────────────

def load_skill(task_scores: dict, prompt: str = "") -> tuple[str, str]:
    """
    Load the most relevant skill file as system context.
    Returns: (system_prompt, skill_name_loaded)

    Always loads base context (pipeline-context.md, truncated).
    Then loads the dominant task skill on top.
    """
    system_parts = []
    skill_loaded = "base-only"

    # Always inject base context (machine identity, paths, conventions)
    if BASE_CONTEXT_FILE.exists():
        base_content = BASE_CONTEXT_FILE.read_text()[:BASE_INJECT_LIMIT]
        system_parts.append(base_content)
# Find dominant task using score first, TASK_PRIORITY to break ties
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

            # Inject dynamic context: file manifest, referenced files, KB, memory
            # Sits after skill (behavior) and before constraints (rules).
            context_block = load_context(prompt, dominant_task)
            if context_block:
                system_parts.append(context_block)

 # Inject hard constraints AFTER skill — guaranteed not truncated
            constraint_block = load_constraints(dominant_task)
            if constraint_block:
                system_parts.append(constraint_block)
        else:
            print(f"[WARN] Skill file not found: {skill_path}")
            # Try remaining tasks in score order as fallback
            for task, score in sorted(task_scores.items(), key=lambda x: -x[1]):
                if task == dominant_task:
                    continue
                if task in SKILL_FILES and SKILL_FILES[task].exists():
                    skill_content = SKILL_FILES[task].read_text()[:SKILL_INJECT_LIMIT]
                    system_parts.append(skill_content)
                    skill_loaded = f"{task}(fallback-from-{dominant_task})"
                    print(f"[WARN] Using fallback skill: {task}")
                    break

    return "\n\n---\n\n".join(system_parts), skill_loaded


# ─── Disambiguation ───────────────────────────────────────────────────────────

def build_prompt_with_disambiguation(prompt: str) -> str:
    """
    Prepend any needed disambiguation clarifications to the prompt.
    Keeps the original prompt intact — just adds context before it.
    """
    clarifications = check_disambiguation(prompt)

    if not clarifications:
        return prompt

    clarification_block = "\n".join(f"[CONTEXT] {c}" for c in clarifications)
    return f"{clarification_block}\n\n{prompt}"


# ─── Ollama API ───────────────────────────────────────────────────────────────

def call_ollama(model: str, prompt: str, system: str = "") -> str:
    """Send a single prompt to the inference backend and return response text."""
    try:
        return call_backend(model, prompt, stream=False, temperature=0.7, system=system)
    except BackendConnectionError as e:
        if isinstance(e.original, requests.exceptions.Timeout):
            print("\n[ERROR] Ollama timed out after 120 seconds.")
            print("  → Model may still be loading. Wait 30s and retry.")
        else:
            print("\n[ERROR] Cannot connect to Ollama.")
            print("  → Try: sudo systemctl start ollama")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n[ERROR] Ollama API error: {e}")
        sys.exit(1)


def call_ollama_chat(model: str, messages: list, system: str = "") -> str:
    """
    Send a conversation history to the inference backend and return the response text.
    Used when a thread is active — flattens message history to a single prompt string
    (mirrors build_multi_turn_prompt() in pipeline_api.py).

    system: injected as system context.
    messages: list of {"role": "user"|"assistant", "content": "..."} dicts,
              with the last entry being the current user message.
    """
    if not messages:
        flat_prompt = ""
    elif len(messages) == 1:
        flat_prompt = messages[0]["content"]
    else:
        history = messages[:-1]
        current = messages[-1]["content"]
        lines = []
        for msg in history:
            label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{label}: {msg['content']}")
        history_block = "\n".join(lines)
        flat_prompt = (
            f"[Conversation history]\n{history_block}\n\n"
            f"[Current request]\n{current}"
        )

    try:
        return call_backend(model, flat_prompt, stream=False, temperature=0.7, system=system)
    except BackendConnectionError as e:
        if isinstance(e.original, requests.exceptions.Timeout):
            print("\n[ERROR] Ollama timed out after 120 seconds.")
            print("  → Model may still be loading. Wait 30s and retry.")
        else:
            print("\n[ERROR] Cannot connect to Ollama.")
            print("  → Try: sudo systemctl start ollama")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n[ERROR] Ollama API error: {e}")
        sys.exit(1)


# ─── Logging ──────────────────────────────────────────────────────────────────

def log_interaction(
    prompt: str,
    selected_model: str,
    task_scores: dict,
    model_scores: dict,
    skill_loaded: str,
    output: str,
    disambiguated: bool,
    retried: bool = False,
    router_path: str = "default",
    eval_passed: bool | None = None,
    eval_score: int | None = None,
    eval_failures: list | None = None,
):
    """
    Append one JSON line to the routing log file.
    Each line is a complete, self-contained JSON object.
    Use jsonl format (one JSON object per line) for easy grep and parsing.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "ts":            datetime.now().isoformat(),
        "prompt":        prompt[:200],          # truncate long prompts
        "model":         selected_model,
        "router_path":   router_path,           # "keyword", "llm", or "default"
        "task_scores":   {k: v for k, v in task_scores.items() if v > 0},
        "model_scores":  {k: round(v, 2) for k, v in model_scores.items()},
        "skill_loaded":  skill_loaded,
        "disambiguated": disambiguated,
        "constraint_injected": skill_loaded if load_constraints(skill_loaded) else False,
        "output_len":    len(output),
        "output_preview": output[:300],        # first 300 chars for review
        "correct":       None,                 # fill in manually or via evaluator later
        "notes":         "",                   # fill in manually after review
        "retried":       retried,              # true if this entry is a retry attempt
        "eval_passed":   eval_passed,
        "eval_score":    eval_score,
        "eval_failures": eval_failures,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ─── Debug Output ─────────────────────────────────────────────────────────────

def print_debug(task_scores, model_scores, selected_model, skill_loaded, disambiguated):
    print("\n" + "─" * 52)
    print("  ROUTING DEBUG")
    print("─" * 52)

    active = {k: v for k, v in task_scores.items() if v > 0}
    if active:
        print("  Task signals detected:")
        for task, score in sorted(active.items(), key=lambda x: -x[1]):
            bar = "█" * score
            print(f"    {task:<15} {bar} ({score})")
    else:
        print("  Task signals: none → default model")

    print()
    print("  Model scores:")
    for model, score in sorted(model_scores.items(), key=lambda x: -x[1]):
        marker = " ← selected" if model == selected_model else ""
        print(f"    {model:<25} {score:.2f}{marker}")

    print()
    print(f"  [CONTEXT] Skill loaded: {skill_loaded}")

    if disambiguated:
        print(f"  [DISAMBIG] Clarification injected into prompt")

    print("─" * 52 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    _ollama_base = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

    parser = argparse.ArgumentParser(
        description="pipeline AI pipeline"
    )
    parser.add_argument("prompt", nargs="?", help="Prompt to send")
    parser.add_argument("--model",       help="Override model selection")
    parser.add_argument("--debug",       action="store_true")
    parser.add_argument("--no-log",      action="store_true",
                        help="Skip writing to routing log")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--show-system", action="store_true",
                        help="Print assembled system prompt and exit without calling Ollama")
    parser.add_argument("--new-thread",  action="store_true",
                        help="Start a new conversation thread and print its ID")
    parser.add_argument("--thread",      metavar="THREAD_ID",
                        help="Continue an existing conversation thread by ID")
    parser.add_argument("--list-threads", action="store_true",
                        help="List recent conversation threads and exit")
    parser.add_argument("--close-thread", metavar="THREAD_ID",
                        help="Summarize and archive a thread, then exit")
    args = parser.parse_args()

    # Handle --close-thread
    if args.close_thread:
        try:
            out_path = close_thread(args.close_thread)
            print(f"[close-thread] Summary written to: {out_path}")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        sys.exit(0)

    # Handle --list-threads
    if args.list_threads:
        print_threads()
        sys.exit(0)

    # Handle --list-models
    if args.list_models:
        try:
            r = requests.get(f"{_ollama_base}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            print("Available models:")
            for m in models:
                print(f"  {m}")
        except Exception:
            print("[ERROR] Cannot reach Ollama")
        sys.exit(0)

    # Get prompt
    if not args.prompt:
        if not sys.stdin.isatty():
            args.prompt = sys.stdin.read().strip()
        else:
            parser.print_help()
            sys.exit(1)

    prompt = args.prompt

    # Route
    if args.model:
        selected_model = args.model
        task_scores    = {}
        model_scores   = {args.model: 1.0}
        router_path    = "default"
        skill_loaded   = "override"
        system_context = ""
        disambiguated  = False
    else:
        selected_model, task_scores, model_scores, router_path = route(prompt)

        # Load skill context
        system_context, skill_loaded = load_skill(task_scores, prompt)

        # Disambiguation
        clarifications = check_disambiguation(prompt)
        disambiguated  = len(clarifications) > 0
        prompt_to_send = build_prompt_with_disambiguation(prompt)

    if args.model:
        prompt_to_send = prompt

    # Debug output
    if args.debug and not args.model:
        print_debug(task_scores, model_scores, selected_model,
                    skill_loaded, disambiguated)

    # Status line (always shown)
    ts = datetime.now().strftime("%H:%M:%S")
    context_indicator = f"[skill:{skill_loaded}]" if not args.model else "[override]"
    print(f"[{ts}] → {selected_model} {context_indicator}\n")

    # --show-system: print assembled system prompt and exit before any API call
    if args.show_system:
        print(system_context)
        sys.exit(0)

    # ── Thread setup ──────────────────────────────────────────────────────────
    thread = None

    if args.new_thread:
        thread = new_thread()
        print(f"[thread:{thread['thread_id']}] New thread started.\n")

    elif args.thread:
        thread = load_thread(args.thread)
        if thread is None:
            print(f"[WARN] Thread not found: {args.thread}. Starting fresh (no history).")
        else:
            turn_count = len(thread["messages"]) // 2
            print(f"[thread:{thread['thread_id']}] Resuming ({turn_count} prior turn(s)).\n")

    # ── Call Ollama ───────────────────────────────────────────────────────────
    if thread is not None:
        # Chat path — passes full conversation history for continuity
        history = get_thread_as_messages(thread)
        history.append({"role": "user", "content": prompt_to_send})
        output = call_ollama_chat(selected_model, history, system=system_context)
    else:
        # Stateless path — original single-shot behaviour
        output = call_ollama(selected_model, prompt_to_send, system=system_context)

    # Evaluate output — retry once on critical failure, skip on --model override
    retried = False
    eval_result = None
    if not args.model:
        eval_result = evaluate(output, skill_loaded)
        if not eval_result.passed:
            critical_issues = [f.issue for f in eval_result.failures if f.severity == "critical"]
            if critical_issues:
                # Log the failed first attempt before retrying
                if not args.no_log:
                    log_interaction(
                        prompt         = prompt,
                        selected_model = selected_model,
                        task_scores    = task_scores,
                        model_scores   = model_scores,
                        skill_loaded   = skill_loaded,
                        output         = output,
                        disambiguated  = disambiguated,
                        retried        = False,
                        router_path    = router_path,
                        eval_passed    = eval_result.passed,
                        eval_score     = round(eval_result.score * 100),
                        eval_failures  = [{"type": f.type, "severity": f.severity, "issue": f.issue} for f in eval_result.failures],
                    )
                # Build retry prompt with specific failure reasons
                issues_text = "\n".join(f"- {issue}" for issue in critical_issues)
                retry_prompt = (
                    f"[EVAL RETRY] Your previous response had these issues:\n"
                    f"{issues_text}\n\n"
                    f"{prompt_to_send}"
                )
                if thread is not None:
                    history[-1] = {"role": "user", "content": retry_prompt}
                    output = call_ollama_chat(selected_model, history, system=system_context)
                else:
                    output = call_ollama(selected_model, retry_prompt, system=system_context)
                retried = True
                eval_result = evaluate(output, skill_loaded)

    # Print final output only — one print, whether or not retry fired
    if retried:
        print(f"[RETRY]\n")
    print(output)
    print()

    # ── Save thread ───────────────────────────────────────────────────────────
    if thread is not None:
        thread = append_exchange(thread, prompt_to_send, output)
        save_thread(thread)
        turn_count = len(thread["messages"]) // 2
        print(f"[thread:{thread['thread_id']}] Turn {turn_count} saved. "
              f"Use --thread {thread['thread_id']} to continue.")

    # Log final output (retry attempt if retried, only attempt otherwise)
    if not args.no_log:
        log_interaction(
            prompt         = prompt,
            selected_model = selected_model,
            task_scores    = task_scores,
            model_scores   = model_scores,
            skill_loaded   = skill_loaded,
            output         = output,
            disambiguated  = disambiguated,
            retried        = retried,
            router_path    = router_path,
            eval_passed    = eval_result.passed if eval_result else None,
            eval_score     = round(eval_result.score * 100) if eval_result else None,
            eval_failures  = [{"type": f.type, "severity": f.severity, "issue": f.issue} for f in eval_result.failures] if eval_result else None,
        )


if __name__ == "__main__":
    main()
