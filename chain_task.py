#!/usr/bin/env python3
# chain_task.py — 3-step chained pipeline runner for the pipeline AI pipeline
# v2 — eval fields added to JSONL log entries
#
# Changes from v1:
#   - log_chain_interaction() gains eval_passed (bool|null), eval_score
#     (int|null), and eval_failures (list[{type,severity,issue}]|null).
#   - run_step() passes eval outcome from evaluate() to log call.
#
# What this file does:
#   Accepts a list of step prompts and runs them in sequence through the
#   existing pipeline. The output of step N is automatically prepended to
#   the prompt of step N+1 as context (capped at 1500 chars). Each step
#   uses the full route → skill → context → constraints → Ollama → evaluate
#   pipeline from run_task.py. All steps are logged to routing-log.jsonl
#   with a shared chain_id for identification.
#
# CLI flags:
#   --chain "step1" "step2" "step3"   Ordered list of prompts to run in sequence
#   --save-to filename.md             Save all outputs to vault/AI/memory/FILENAME
#   --dry-run                         Print routing info, skip Ollama calls
#
# Dependencies:
#   router.py       route(), check_disambiguation()
#   run_task.py     call_ollama(), load_skill(), build_prompt_with_disambiguation()
#   config.py       LOG_FILE, VAULT
#   evaluator.py    evaluate(), RULE_SETS
#   constraints.py  load_constraints()

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

from router import route, check_disambiguation
from config import LOG_FILE, VAULT
from evaluator import evaluate, RULE_SETS
from constraints import load_constraints
from run_task import (
    call_ollama,
    load_skill,
    build_prompt_with_disambiguation,
)

CHAIN_OUTPUT_DIR = VAULT / "AI/memory"
CONTEXT_CAP = 1500


# ─── Logging ──────────────────────────────────────────────────────────────────

def log_chain_interaction(
    prompt: str,
    selected_model: str,
    task_scores: dict,
    model_scores: dict,
    skill_loaded: str,
    output: str,
    disambiguated: bool,
    retried: bool,
    router_path: str,
    chain_id: str,
    step_num: int,
    eval_passed: bool | None = None,
    eval_score: int | None = None,
    eval_failures: list | None = None,
):
    """Append a JSONL log entry. Includes chain_id and chain_step fields."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "ts":                datetime.now().isoformat(),
        "prompt":            prompt[:200],
        "model":             selected_model,
        "router_path":       router_path,
        "task_scores":       {k: v for k, v in task_scores.items() if v > 0},
        "model_scores":      {k: round(v, 2) for k, v in model_scores.items()},
        "skill_loaded":      skill_loaded,
        "disambiguated":     disambiguated,
        "constraint_injected": skill_loaded if load_constraints(skill_loaded) else False,
        "output_len":        len(output),
        "output_preview":    output[:300],
        "correct":           None,
        "notes":             "",
        "retried":           retried,
        "chain_id":          chain_id,
        "chain_step":        step_num,
        "eval_passed":       eval_passed,
        "eval_score":        eval_score,
        "eval_failures":     eval_failures,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ─── Context Injection ────────────────────────────────────────────────────────

def inject_previous_output(prompt: str, previous_output: str, prev_step_num: int) -> str:
    """Prepend previous step output as a labeled context block."""
    capped = previous_output[:CONTEXT_CAP]
    if len(previous_output) > CONTEXT_CAP:
        capped += " [truncated]"

    context_block = (
        f"--- Output from step {prev_step_num} ---\n"
        f"{capped}\n"
        f"--- End step {prev_step_num} output ---\n\n"
    )
    return context_block + prompt


# ─── Step Runner ──────────────────────────────────────────────────────────────

def run_step(
    prompt: str,
    step_num: int,
    total_steps: int,
    chain_id: str,
    dry_run: bool = False,
) -> tuple[str, bool]:
    """
    Run a single pipeline step through the full route → skill → Ollama → evaluate
    sequence. Returns (output, eval_passed). On dry-run returns ("", True).
    """
    selected_model, task_scores, model_scores, router_path = route(prompt)
    system_context, skill_loaded = load_skill(task_scores, prompt)
    clarifications = check_disambiguation(prompt)
    disambiguated = len(clarifications) > 0
    prompt_to_send = build_prompt_with_disambiguation(prompt)

    # Determine dominant task for display
    active = {k: v for k, v in task_scores.items() if v > 0}
    if active:
        dominant_task = max(active, key=active.get)
        dominant_score = active[dominant_task]
    else:
        dominant_task = "default"
        dominant_score = 0

    print(f"[Step {step_num}/{total_steps}] Routing: {dominant_task} "
          f"(score: {dominant_score}) → {selected_model}")

    if dry_run:
        print(f"[Step {step_num}/{total_steps}] [DRY RUN] Skipping Ollama call.")
        return "", True

    output = call_ollama(selected_model, prompt_to_send, system=system_context)

    # Evaluate
    eval_result = evaluate(output, skill_loaded)
    total_rules = len(RULE_SETS.get(skill_loaded, RULE_SETS["default"]))
    pass_count = total_rules - len(eval_result.failures)
    status = "PASS" if eval_result.passed else "FAIL"
    print(f"[Step {step_num}/{total_steps}] Evaluator: {status} "
          f"(score {pass_count}/{total_rules})")

    if not eval_result.passed:
        for f in eval_result.failures:
            severity_tag = "[CRITICAL]" if f.severity == "critical" else "[WARN]"
            print(f"  {severity_tag} {f.issue}")

    print(f"[Step {step_num}/{total_steps}] Done. ({len(output)} chars output)")

    log_chain_interaction(
        prompt=prompt,
        selected_model=selected_model,
        task_scores=task_scores,
        model_scores=model_scores,
        skill_loaded=skill_loaded,
        output=output,
        disambiguated=disambiguated,
        retried=False,
        router_path=router_path,
        chain_id=chain_id,
        step_num=step_num,
        eval_passed=eval_result.passed,
        eval_score=round(eval_result.score * 100),
        eval_failures=[{"type": f.type, "severity": f.severity, "issue": f.issue} for f in eval_result.failures],
    )

    return output, eval_result.passed


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="pipeline chained pipeline runner")
    parser.add_argument(
        "--chain", nargs="+", required=True, metavar="STEP",
        help="Ordered list of step prompts to run in sequence"
    )
    parser.add_argument(
        "--save-to", metavar="FILENAME",
        help="Save all outputs to vault/AI/memory/FILENAME"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print routing info without calling Ollama"
    )
    args = parser.parse_args()

    original_prompts = args.chain
    total = len(original_prompts)
    chain_id = datetime.now().strftime("%Y%m%dT%H%M%S")

    outputs = []
    eval_passes = 0

    for i, base_prompt in enumerate(original_prompts, start=1):
        # Inject previous step output as context
        if i > 1 and outputs:
            step_prompt = inject_previous_output(base_prompt, outputs[-1], i - 1)
        else:
            step_prompt = base_prompt

        output, passed = run_step(
            prompt=step_prompt,
            step_num=i,
            total_steps=total,
            chain_id=chain_id,
            dry_run=args.dry_run,
        )
        outputs.append(output)
        if passed:
            eval_passes += 1

    print(f"\n=== Chain complete: {eval_passes}/{total} steps passed ===")

    if args.save_to and not args.dry_run:
        save_path = CHAIN_OUTPUT_DIR / args.save_to
        save_path.parent.mkdir(parents=True, exist_ok=True)

        sections = []
        for i, (orig_prompt, output) in enumerate(zip(original_prompts, outputs), start=1):
            header = f"## Step {i}: {orig_prompt[:80]}"
            sections.append(f"{header}\n\n{output}")

        save_path.write_text("\n\n".join(sections) + "\n")
        print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
