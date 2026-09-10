#!/usr/bin/env python3
# orchestrator.py — Goal decomposition and autonomous multi-step execution
# v3
#
# Changes from v2:
#   - All prior step outputs now injected into each pipeline step, not just the
#     immediately preceding one. Steps before the most recent are condensed to
#     SUMMARY_CAP (400 chars) to preserve full chain context without overflowing
#     the model's usable context window. The most recent step retains CONTEXT_CAP
#     (2000 chars). For a 3-step chain, step 3 now sees step 1 summary + step 2
#     full output instead of step 2 only — prevents context loss mid-chain.
#   - SUMMARY_CAP constant added alongside CONTEXT_CAP.
#   - format="json" added to decompose_goal() Ollama payload. Constrains model
#     output to valid JSON natively — eliminates the regex fallback as the primary
#     failure mode.
#
# Changes from v1:
#   - --plan-only flag added: decompose goal, print plan, exit without executing.
#     Safety gate for agentic operation — inspect before committing.
#   - Per-step retry added to run_step(): if a pipeline step fails critical eval,
#     retry once with failure reasons prepended (mirrors run_task.py behaviour).
#   - Prior session context injected into decompose_goal(): rag_index.query_memory()
#     surfaces relevant past session memory before the decomposer runs.
#
# What this file does:
#   Accepts a high-level goal, uses the primary model to decompose it into a typed
#   step plan (JSON), executes each step autonomously, chains outputs, and writes
#   a memory file on completion — no manual intervention required between goal
#   input and final output.
#
# Step types in a plan:
#   pipeline  — routes through the full local AI stack (route → skill → context
#                → constraints → Ollama → evaluate). Output feeds next step.
#   tool      — calls a registered tool from tools.py (read_vault_file,
#                write_vault_file). Tool output is injected into the next
#                pipeline step. web_search was removed in tools.py v4 — there
#                is no outbound-network tool on this path.
#
# CLI usage:
#   python orchestrator.py "Research QLoRA fine-tuning and write a summary"
#   python orchestrator.py "your goal" --dry-run
#   python orchestrator.py "your goal" --save-to output.md
#   python orchestrator.py "your goal" --max-steps 5
#   python orchestrator.py "your goal" --no-memory   # skip auto memory write

import os
import sys
import re
import json
import argparse
import requests
from datetime import datetime
from pathlib import Path

from router import route, check_disambiguation
from config import OLLAMA_URL, VAULT, LOG_FILE
from run_task import call_ollama, load_skill, build_prompt_with_disambiguation
from constraints import load_constraints
from evaluator import evaluate, RULE_SETS
from tools import call_tool, TOOL_REGISTRY
import rag_index

MEMORY_DIR    = VAULT / "AI/memory"
CONTEXT_CAP   = 2000   # chars for the most recent prior step
SUMMARY_CAP   = 400    # chars per earlier step (condensed — all steps but last)
MAX_STEPS_CAP = 8      # hard ceiling regardless of --max-steps

_DECOMPOSE_MODEL = os.getenv("PRIMARY_MODEL", "qwen2.5:14b")
_DECOMPOSE_TEMP  = 0.1   # low temperature for reliable JSON

_DECOMPOSE_PROMPT = """\
You are a task planner for a local AI pipeline. Your job is to break the following goal into 2-5 concrete, ordered steps.
{prior_context}
Goal: {goal}

Available step types:
  "pipeline" — routes the prompt through the local AI stack, which has access to the
               knowledge base, skill files, memory, and all local system context.
               Use this for: explaining concepts, answering questions about the pipeline,
               writing code, analysis, summaries, reasoning, or anything this system knows.

  "tool"     — calls a local function directly. Use for vault file operations only.

Available tools (for type="tool" steps only):
  read_vault_file(relative_path)           — read a specific file from the local vault
  write_vault_file(relative_path, content) — write a file to the local vault

Output ONLY valid JSON. No explanation. No markdown. No code fences. Exactly this format:

{{
  "steps": [
    {{"id": 1, "type": "pipeline", "prompt": "specific instruction for this step"}},
    {{"id": 2, "type": "tool", "tool": "read_vault_file", "args": {{"relative_path": "AI/memory/example.md"}}}},
    {{"id": 3, "type": "pipeline", "prompt": "use the file content above to write a summary"}}
  ]
}}

Rules:
- 2 to 5 steps. Never more than 5.
- Every step must have "id" (integer) and "type" (string).
- Pipeline steps must have "prompt" (string) — be specific and self-contained.
- Tool steps must have "tool" (string) and "args" (object).
- The last step must always be type "pipeline" — the final output comes from the AI.
- Default to pipeline. Tool steps are only needed when a specific vault file must be read or written.
- Do not add any text before or after the JSON object.
"""


# ─── Goal Decomposer ──────────────────────────────────────────────────────────

def decompose_goal(goal: str, max_steps: int = 5) -> list[dict]:
    """
    Use the primary model to decompose a high-level goal into a typed step plan.

    Returns a list of step dicts. Each dict has at minimum:
      {"id": int, "type": "pipeline"|"tool", ...}

    Falls back to a single pipeline step (run the goal directly) if:
      - Ollama is unreachable
      - JSON parse fails after cleanup attempts
      - Returned steps list is empty
    """
    # Inject relevant prior session context so the decomposer builds on past work
    prior_context = ""
    try:
        mem_results = rag_index.query_memory(goal, top_k=2)
        if mem_results:
            snippets = []
            for stem, content, score in mem_results:
                if score > 0.25:
                    snippets.append(f"[{stem}]\n{content[:400]}")
            if snippets:
                prior_context = (
                    "\nPrior session context (use to avoid repeating completed work):\n"
                    + "\n\n".join(snippets)
                    + "\n"
                )
    except Exception:
        pass  # RAG failure is non-fatal — decomposer runs without context

    prompt = _DECOMPOSE_PROMPT.format(goal=goal, prior_context=prior_context)
    payload = {
        "model":   _DECOMPOSE_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "format":  "json",
        "options": {"temperature": _DECOMPOSE_TEMP, "num_predict": 1024},
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        raw = r.json().get("response", "").strip()
    except Exception as e:
        print(f"[orchestrator] Decomposer unavailable ({e}) — running goal as single step.")
        return [{"id": 1, "type": "pipeline", "prompt": goal}]

    # Parse JSON — try several cleanup strategies
    steps = _parse_steps_json(raw, goal)
    if not steps:
        print("[orchestrator] Could not parse decomposer output — running goal as single step.")
        return [{"id": 1, "type": "pipeline", "prompt": goal}]

    # Enforce cap
    return steps[:max_steps]


def _parse_steps_json(raw: str, goal: str) -> list[dict]:
    """
    Try to extract a valid steps list from raw model output.
    Attempts:
      1. Direct json.loads()
      2. Extract first {...} block with regex
      3. Give up — return []
    """
    # Attempt 1: direct parse
    try:
        data = json.loads(raw)
        steps = data.get("steps", [])
        if steps and isinstance(steps, list):
            return steps
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract first JSON object from the raw string
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            steps = data.get("steps", [])
            if steps and isinstance(steps, list):
                return steps
        except json.JSONDecodeError:
            pass

    return []


# ─── Step Executor ────────────────────────────────────────────────────────────

def _build_step_prompt(base_prompt: str, prior_outputs: list[tuple[int, str, str]]) -> str:
    """
    Build the full prompt for a pipeline step by injecting all prior step outputs.

    prior_outputs: list of (step_id, step_type, output_text) — most recent last.

    All steps before the most recent are condensed to SUMMARY_CAP chars so the
    full chain context is visible without overflowing usable context. The most
    recent step is injected at CONTEXT_CAP chars (full detail).
    """
    if not prior_outputs:
        return base_prompt

    context_blocks = []

    # Earlier steps — condensed to SUMMARY_CAP so the full chain is visible
    for step_id, step_type, output in prior_outputs[:-1]:
        capped = output[:SUMMARY_CAP]
        if len(output) > SUMMARY_CAP:
            capped += " [truncated]"
        label = f"Step {step_id} output" if step_type == "pipeline" else f"Tool result (step {step_id})"
        context_blocks.append(f"--- {label} (summary) ---\n{capped}\n--- end ---")

    # Most recent step — full CONTEXT_CAP
    prev_id, prev_type, prev_output = prior_outputs[-1]
    capped = prev_output[:CONTEXT_CAP]
    if len(prev_output) > CONTEXT_CAP:
        capped += " [truncated]"
    label = f"Step {prev_id} output" if prev_type == "pipeline" else f"Tool result (step {prev_id})"
    context_blocks.append(f"--- {label} ---\n{capped}\n--- end ---")

    return "\n\n".join(context_blocks) + "\n\n" + base_prompt


def run_step(
    step: dict,
    step_num: int,
    total_steps: int,
    prior_outputs: list[tuple[int, str, str]],
    dry_run: bool = False,
) -> tuple[str, bool]:
    """
    Execute a single step. Returns (output_text, eval_passed).

    Pipeline steps route through the full local AI stack.
    Tool steps call the registered tool directly — no Ollama call.
    """
    step_type = step.get("type", "pipeline")
    step_id   = step.get("id", step_num)

    # ── Tool step ─────────────────────────────────────────────────────────────
    if step_type == "tool":
        tool_name = step.get("tool", "")
        tool_args = step.get("args", {})

        if tool_name not in TOOL_REGISTRY:
            known = ", ".join(TOOL_REGISTRY)
            output = f"[orchestrator] Unknown tool: {tool_name!r}. Known: {known}"
            print(f"  [Step {step_num}/{total_steps}] tool:{tool_name} → ERROR")
            return output, False

        print(f"  [Step {step_num}/{total_steps}] tool:{tool_name} args={tool_args}")
        if dry_run:
            print(f"  [DRY RUN] Skipping tool call.")
            return f"[DRY RUN] tool:{tool_name}", True

        output = call_tool(tool_name, tool_args)
        print(f"  [Step {step_num}/{total_steps}] tool:{tool_name} → {len(output)} chars")
        return output, True

    # ── Pipeline step ─────────────────────────────────────────────────────────
    base_prompt   = step.get("prompt", "")
    step_prompt   = _build_step_prompt(base_prompt, prior_outputs)

    selected_model, task_scores, model_scores, router_path = route(step_prompt)
    system_context, skill_loaded = load_skill(task_scores, step_prompt)
    prompt_to_send = build_prompt_with_disambiguation(step_prompt)

    # Determine dominant task for display
    active = {k: v for k, v in task_scores.items() if v > 0}
    dominant = max(active, key=active.get) if active else "default"

    print(f"  [Step {step_num}/{total_steps}] pipeline → {selected_model} "
          f"[{dominant}/{router_path}] skill:{skill_loaded}")

    if dry_run:
        print(f"  [DRY RUN] Skipping Ollama call.")
        return f"[DRY RUN] pipeline step {step_num}", True

    output = call_ollama(selected_model, prompt_to_send, system=system_context)

    # Evaluate
    eval_result  = evaluate(output, skill_loaded)
    total_rules  = len(RULE_SETS.get(skill_loaded, RULE_SETS["default"]))
    pass_count   = total_rules - len(eval_result.failures)
    status       = "PASS" if eval_result.passed else "FAIL"
    print(f"  [Step {step_num}/{total_steps}] eval:{status} ({pass_count}/{total_rules}) "
          f"→ {len(output)} chars output")

    if not eval_result.passed:
        for f in eval_result.failures:
            tag = "[CRITICAL]" if f.severity == "critical" else "[WARN]"
            print(f"    {tag} {f.issue}")

        # Retry once on critical failure — bad output must not feed the next step
        critical = [f.issue for f in eval_result.failures if f.severity == "critical"]
        if critical:
            issues_text = "\n".join(f"- {issue}" for issue in critical)
            retry_prompt = (
                f"[EVAL RETRY] Your previous response had these issues:\n"
                f"{issues_text}\n\n"
                f"{prompt_to_send}"
            )
            output = call_ollama(selected_model, retry_prompt, system=system_context)
            eval_result = evaluate(output, skill_loaded)
            retry_status = "PASS" if eval_result.passed else "FAIL"
            print(f"  [Step {step_num}/{total_steps}] retry → eval:{retry_status} "
                  f"→ {len(output)} chars")

    return output, eval_result.passed


# ─── Memory Writer ────────────────────────────────────────────────────────────

def write_orchestrator_memory(
    goal: str,
    steps: list[dict],
    outputs: list[tuple[int, str, str]],
    session_id: str,
) -> Path:
    """
    Write a structured memory file for the orchestrator session.
    Format mirrors thread summaries — YAML frontmatter + markdown body.
    Rebuilds RAG memory index so the file is immediately queryable.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    steps_run = len(outputs)

    # Build body
    sections = [f"# Orchestrator Session — {today}\n\n**Goal:** {goal}\n"]
    for step_id, step_type, output in outputs:
        step_dict = next((s for s in steps if s.get("id") == step_id), {})
        label = step_dict.get("prompt") or f"{step_type}:{step_dict.get('tool', '')}"
        sections.append(f"## Step {step_id} ({step_type}): {label[:80]}\n\n{output}\n")

    body = "\n".join(sections)

    frontmatter = (
        f"---\n"
        f"date: {today}\n"
        f"session_id: {session_id}\n"
        f"goal: {json.dumps(goal)}\n"
        f"steps_run: {steps_run}\n"
        f"type: orchestrator-session\n"
        f"---\n\n"
    )

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MEMORY_DIR / f"{session_id}-orchestrator.md"
    out_path.write_text(frontmatter + body)

    # Rebuild memory index so this session is immediately queryable
    rag_index.rebuild_memory()

    return out_path


# ─── Main Orchestrator ────────────────────────────────────────────────────────

def run_orchestrator(
    goal: str,
    max_steps: int = 5,
    save_to: str | None = None,
    dry_run: bool = False,
    write_memory: bool = True,
) -> str:
    """
    Full orchestration loop:
      1. Decompose goal → typed step plan
      2. Execute steps in order, chaining outputs
      3. Write memory file (unless --no-memory)
      4. Return final output text

    Args:
        goal:         High-level goal string
        max_steps:    Maximum steps to execute (capped at MAX_STEPS_CAP)
        save_to:      If set, write full output to vault/AI/memory/FILENAME
        dry_run:      Route and print plan without calling Ollama or tools
        write_memory: Write a memory file on completion
    """
    session_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    max_steps  = min(max_steps, MAX_STEPS_CAP)

    print(f"\n[orchestrator] session:{session_id}")
    print(f"[orchestrator] goal: {goal[:100]}")
    print(f"[orchestrator] decomposing into steps...\n")

    steps = decompose_goal(goal, max_steps)

    print(f"[orchestrator] plan: {len(steps)} step(s)")
    for s in steps:
        if s["type"] == "pipeline":
            print(f"  step {s['id']}: pipeline — {s.get('prompt', '')[:70]}")
        else:
            print(f"  step {s['id']}: tool:{s.get('tool','')} args={s.get('args',{})}")
    print()

    prior_outputs: list[tuple[int, str, str]] = []
    eval_passes = 0

    for i, step in enumerate(steps, start=1):
        output, passed = run_step(
            step=step,
            step_num=i,
            total_steps=len(steps),
            prior_outputs=prior_outputs,
            dry_run=dry_run,
        )
        prior_outputs.append((step.get("id", i), step.get("type", "pipeline"), output))
        if passed:
            eval_passes += 1

    print(f"\n[orchestrator] complete: {eval_passes}/{len(steps)} steps passed eval")

    # Final output = last pipeline step's output
    final_output = ""
    for _, stype, output in reversed(prior_outputs):
        if stype == "pipeline":
            final_output = output
            break
    if not final_output and prior_outputs:
        _, _, final_output = prior_outputs[-1]

    # Save to file if requested
    if save_to and not dry_run:
        sections = []
        for step_id, step_type, output in prior_outputs:
            step_dict = next((s for s in steps if s.get("id") == step_id), {})
            label = step_dict.get("prompt") or f"tool:{step_dict.get('tool','')}"
            sections.append(f"## Step {step_id} ({step_type}): {label[:80]}\n\n{output}")
        file_content = "\n\n".join(sections) + "\n"

        save_path = MEMORY_DIR / save_to
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(file_content)
        print(f"[orchestrator] saved to: {save_path}")

    # Write memory
    if write_memory and not dry_run:
        mem_path = write_orchestrator_memory(goal, steps, prior_outputs, session_id)
        print(f"[orchestrator] memory written: {mem_path}")

    return final_output


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="local-orchestrator — autonomous multi-step goal execution"
    )
    parser.add_argument("goal", help="High-level goal for the orchestrator to execute")
    parser.add_argument(
        "--max-steps", type=int, default=5, metavar="N",
        help=f"Maximum steps to plan and run (default: 5, hard cap: {MAX_STEPS_CAP})"
    )
    parser.add_argument(
        "--save-to", metavar="FILENAME",
        help="Save all step outputs to vault/AI/memory/FILENAME"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and routing without calling Ollama or tools"
    )
    parser.add_argument(
        "--no-memory", action="store_true",
        help="Skip writing a memory file on completion"
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Decompose goal and print plan without executing — inspect before committing"
    )
    args = parser.parse_args()

    # --plan-only: show the decomposed plan and exit — safety gate for agentic use
    if args.plan_only:
        goal = args.goal
        max_steps = min(args.max_steps, MAX_STEPS_CAP)
        print(f"\n[orchestrator] --plan-only  goal: {goal[:100]}")
        print("[orchestrator] decomposing...\n")
        steps = decompose_goal(goal, max_steps)
        print(f"[plan] {len(steps)} step(s)\n")
        for s in steps:
            sid = s.get("id", "?")
            if s.get("type") == "pipeline":
                print(f"  step {sid}: pipeline")
                print(f"    prompt: {s.get('prompt', '')}")
            else:
                print(f"  step {sid}: tool:{s.get('tool', '')}")
                print(f"    args:   {s.get('args', {})}")
            print()
        print("Run without --plan-only to execute.")
        sys.exit(0)

    output = run_orchestrator(
        goal        = args.goal,
        max_steps   = args.max_steps,
        save_to     = args.save_to,
        dry_run     = args.dry_run,
        write_memory= not args.no_memory,
    )

    print("\n" + "─" * 60)
    print("[orchestrator] Final output:\n")
    print(output)
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
