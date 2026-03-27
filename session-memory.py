#!/usr/bin/env python3
# session-memory.py — Automation script for generating session memory files
# v2
#
# Changes from v1:
#   - format_entries_for_prompt() now extracts the `retried` field from log
#     entries and includes it in the formatted block output when True, so
#     session memory summaries reflect when answers were produced via retry
#     rather than first attempt.
#
# Reads today's routing-log.jsonl entries, calls the primary model to generate
# a structured session memory file, writes it to AI/memory/, and git commits.
#
# Usage:
#   python3 session-memory.py                        # generate for today
#   python3 session-memory.py --date 2026-03-21      # generate for a past date
#   python3 session-memory.py --append               # append to existing file
#   python3 session-memory.py --no-commit            # skip git commit
#   python3 session-memory.py --dry-run              # print output, write nothing
#
# Paths (from config.py — do not hardcode):
#   Log:     VAULT/AI/memory/routing-log.jsonl
#   Output:  VAULT/AI/memory/YYYY-MM-DD-session.md
#
# Hard constraints:
#   - Do NOT modify run_task.py, router.py, or any pipeline file
#   - Memory output dir is VAULT / "AI/memory" — no other location
#   - Session boundary is date prefix on the "ts" field

import os
import sys
import json
import argparse
import subprocess
import requests
from datetime import date, datetime
from pathlib import Path

# ─── Paths (match config.py exactly) ─────────────────────────────────────────

VAULT      = Path(os.getenv("PIPELINE_VAULT_PATH", "./vault"))
MEMORY_DIR = VAULT / "AI/memory"
LOG_FILE   = MEMORY_DIR / "routing-log.jsonl"
VAULT_ROOT = Path(os.getenv("PIPELINE_STORAGE_PATH", "./storage"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434") + "/api/generate"
MODEL      = os.getenv("PRIMARY_MODEL", "qwen2.5:14b")


# ─── Log Reading ──────────────────────────────────────────────────────────────

def load_todays_entries(target_date: str) -> list[dict]:
    """
    Read routing-log.jsonl and return only entries whose 'ts' field
    starts with target_date (format: YYYY-MM-DD).

    Returns empty list if log is missing or no entries match.
    """
    if not LOG_FILE.exists():
        print(f"[WARN] Log file not found: {LOG_FILE}")
        return []

    entries = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("ts", "").startswith(target_date):
                    entries.append(entry)
            except json.JSONDecodeError:
                continue

    return entries


# ─── Log Summarisation ────────────────────────────────────────────────────────

def format_entries_for_prompt(entries: list[dict]) -> str:
    """
    Convert log entries into a compact, model-readable summary block.
    Keeps only the fields that are useful for memory generation.
    Truncates output_preview to 150 chars to keep the prompt manageable.
    """
    lines = []
    for i, e in enumerate(entries, 1):
        ts_short  = e.get("ts", "")[:16]          # YYYY-MM-DDTHH:MM
        prompt    = e.get("prompt", "")[:120]
        model     = e.get("model", "unknown")
        skill     = e.get("skill_loaded", "none")
        tasks     = e.get("task_scores", {})
        preview   = e.get("output_preview", "")[:150]
        correct   = e.get("correct")
        notes     = e.get("notes", "")
        retried   = e.get("retried", False)

        task_str  = ", ".join(f"{k}:{v}" for k, v in tasks.items()) or "none"
        status    = "unreviewed" if correct is None else ("correct" if correct else "wrong")

        block = (
            f"[{i}] {ts_short}\n"
            f"  prompt:  {prompt}\n"
            f"  model:   {model}  |  skill: {skill}  |  tasks: {task_str}\n"
            f"  status:  {status}"
            + (f'  |  notes: {notes}' if notes else "")
            + (f'  |  retried: true' if retried else "") + "\n"
            f"  output:  {preview}"
        )
        lines.append(block)

    return "\n\n".join(lines)


def extract_session_stats(entries: list[dict]) -> dict:
    """
    Compute simple statistics from log entries for the memory file header.
    """
    models  = {}
    skills  = {}
    tasks   = {}

    for e in entries:
        m = e.get("model", "unknown")
        models[m] = models.get(m, 0) + 1

        s = e.get("skill_loaded", "none")
        skills[s] = skills.get(s, 0) + 1

        for task, score in e.get("task_scores", {}).items():
            if score > 0:
                tasks[task] = tasks.get(task, 0) + 1

    correct   = sum(1 for e in entries if e.get("correct") is True)
    incorrect = sum(1 for e in entries if e.get("correct") is False)

    return {
        "total":     len(entries),
        "models":    models,
        "skills":    skills,
        "tasks":     tasks,
        "correct":   correct,
        "incorrect": incorrect,
    }


# ─── Model Prompt ─────────────────────────────────────────────────────────────

MEMORY_FORMAT_TEMPLATE = """\
---
title: Session Summary — {LONG_DATE}
tags: [ai-pipeline, {TAG_LIST}]
created: {DATE}
status: complete
---

# Session Summary — {LONG_DATE}
## {SUBTITLE}

---

## What Was Accomplished

### 1. [First major task or accomplishment]

**Problem:** [What problem existed or what was needed]

**Fix:** [What was built or changed, specific file names and function names]

[Add more numbered subsections as needed]

---

## Session Statistics

| Metric | Value |
|---|---|
| Total prompts | {TOTAL} |
| Skills used | {SKILLS} |
| Tasks routed | {TASKS} |
| Reviewed correct | {CORRECT} |
| Reviewed incorrect | {INCORRECT} |

---

## Files Changed This Session

| File | Status | Location |
|---|---|---|
| `example.py` | New / Modified | pipeline/ |

---

## What Is Not Yet Built

- List any items still pending based on the session prompts
"""

SKILL_FILE = VAULT / "AI/skills/session-memory.md"

_SYSTEM_PROMPT_FALLBACK = """\
You are writing a structured session memory file for a local AI pipeline project.
Output ONLY the memory file markdown — no preamble, no explanation, no code fences.
"""

def load_system_prompt() -> str:
    """Read system prompt from the session-memory skill file. Falls back to inline if missing."""
    if SKILL_FILE.exists():
        return SKILL_FILE.read_text()
    print(f"[WARN] Skill file not found: {SKILL_FILE}")
    print("  → Using fallback system prompt.")
    return _SYSTEM_PROMPT_FALLBACK


def build_model_prompt(target_date: str, entries: list[dict], stats: dict) -> str:
    """
    Construct the full prompt to send to the primary model.
    Includes the template, session statistics, and formatted log entries.
    """
    long_date = datetime.strptime(target_date, "%Y-%m-%d").strftime("%B %d, %Y").replace(" 0", " ")

    # Fill in the template placeholders for statistics only — model fills the rest
    skills_str = ", ".join(
        f"{k}×{v}" for k, v in sorted(stats["skills"].items(), key=lambda x: -x[1])
    )
    tasks_str = ", ".join(
        f"{k}×{v}" for k, v in sorted(stats["tasks"].items(), key=lambda x: -x[1])
    ) or "none"

    template = MEMORY_FORMAT_TEMPLATE.format(
        DATE      = target_date,
        LONG_DATE = long_date,
        TAG_LIST  = "session",          # model will replace with real tags
        SUBTITLE  = "DESCRIBE_THIS_SESSION",
        TOTAL     = stats["total"],
        SKILLS    = skills_str,
        TASKS     = tasks_str,
        CORRECT   = stats["correct"],
        INCORRECT = stats["incorrect"],
    )

    log_block = format_entries_for_prompt(entries)

    return (
        f"Today's date: {target_date}\n"
        f"Session had {stats['total']} pipeline runs.\n\n"
        f"=== LOG ENTRIES ===\n\n{log_block}\n\n"
        f"=== MEMORY FILE FORMAT TO USE ===\n\n{template}\n\n"
        f"Write the complete memory file for this session now. "
        f"Replace all placeholder text. Use only information present in the log entries."
    )


# ─── Ollama Call ──────────────────────────────────────────────────────────────

def call_ollama(prompt: str) -> str:
    """
    Send prompt to the primary model via Ollama /api/generate.
    Returns the model's response text.
    Exits with error message on connection failure.
    """
    payload = {
        "model":  MODEL,
        "prompt": prompt,
        "system": load_system_prompt(),
        "stream": False,
        "options": {"temperature": 0.3},    # lower temp for structured factual output
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to Ollama.")
        print("  → Try: sudo systemctl start ollama")
        sys.exit(1)

    except requests.exceptions.Timeout:
        print("[ERROR] Ollama timed out after 180 seconds.")
        print("  → Model may still be loading. Wait 30s and retry.")
        sys.exit(1)

    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] Ollama API error: {e}")
        sys.exit(1)


# ─── File Writing ─────────────────────────────────────────────────────────────

def write_memory_file(target_date: str, content: str, append: bool = False) -> Path:
    """
    Write the generated memory content to YYYY-MM-DD-session.md.

    append=True: appends a dated divider + new content to existing file.
    append=False (default): skips if file exists and returns None.

    Returns the Path that was written, or None if skipped.
    """
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MEMORY_DIR / f"{target_date}-session.md"

    if out_path.exists() and not append:
        print(f"[WARN] Memory file already exists: {out_path.name}")
        print("  → Use --append to add to it, or delete it first.")
        return None

    if out_path.exists() and append:
        existing = out_path.read_text()
        now = datetime.now().strftime("%H:%M:%S")
        divider = f"\n\n---\n\n<!-- appended {now} -->\n\n"
        out_path.write_text(existing + divider + content)
        print(f"[OK] Appended to: {out_path}")
    else:
        out_path.write_text(content)
        print(f"[OK] Written: {out_path}")

    return out_path


# ─── Git Commit ───────────────────────────────────────────────────────────────

def git_commit(target_date: str) -> bool:
    """
    Run git add . && git commit from the vault root.
    Uses two separate subprocess calls — no shell=True with &&.
    Returns True on success, False on failure.
    """
    commit_msg = f"Auto memory: session {target_date}"

    try:
        # Stage all changes
        add_result = subprocess.run(
            ["git", "-C", str(VAULT_ROOT), "add", "."],
            capture_output=True, text=True
        )
        if add_result.returncode != 0:
            print(f"[ERROR] git add failed:\n{add_result.stderr.strip()}")
            return False

        # Commit
        commit_result = subprocess.run(
            ["git", "-C", str(VAULT_ROOT), "commit", "-m", commit_msg],
            capture_output=True, text=True
        )

        if commit_result.returncode == 0:
            print(f"[OK] Committed: {commit_msg}")
            # Print the short commit hash if available
            first_line = commit_result.stdout.strip().splitlines()[0]
            if first_line:
                print(f"     {first_line}")
            return True

        # Exit code 1 from git commit = "nothing to commit"
        if "nothing to commit" in commit_result.stdout + commit_result.stderr:
            print("[INFO] Nothing to commit — vault already up to date.")
            return True

        print(f"[ERROR] git commit failed:\n{commit_result.stderr.strip()}")
        return False

    except FileNotFoundError:
        print("[ERROR] git not found in PATH.")
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate session memory file from routing-log.jsonl"
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Date to generate memory for (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing memory file instead of skipping.",
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Skip git commit after writing the file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated content without writing anything.",
    )
    args = parser.parse_args()

    target_date = args.date

    # Validate date format
    try:
        datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError:
        print(f"[ERROR] Invalid date format: {target_date}")
        print("  → Expected: YYYY-MM-DD")
        sys.exit(1)

    print(f"[session-memory] date={target_date}  log={LOG_FILE.name}")
    print()

    # ── Step 1: Load log entries ───────────────────────────────────────────────
    entries = load_todays_entries(target_date)

    if not entries:
        print(f"[WARN] No log entries found for {target_date}.")
        print("  → Nothing to summarise. Exiting.")
        sys.exit(0)

    print(f"[LOG] Found {len(entries)} entries for {target_date}.")

    # ── Step 2: Build model prompt ─────────────────────────────────────────────
    stats  = extract_session_stats(entries)
    prompt = build_model_prompt(target_date, entries, stats)

    # ── Step 3: Call model ─────────────────────────────────────────────────────
    print(f"[MODEL] Calling {MODEL}...")
    content = call_ollama(prompt)

    if not content:
        print("[ERROR] Model returned empty response. Exiting.")
        sys.exit(1)

    # Strip accidental code fences if the model added them
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    # ── Step 4: Dry run ────────────────────────────────────────────────────────
    if args.dry_run:
        print()
        print("─" * 60)
        print("  DRY RUN — nothing written")
        print("─" * 60)
        print(content)
        print("─" * 60)
        sys.exit(0)

    # ── Step 5: Write file ─────────────────────────────────────────────────────
    out_path = write_memory_file(target_date, content, append=args.append)

    if out_path is None:
        # File existed and --append was not passed — already warned the user
        sys.exit(0)

    # ── Step 6: Git commit ─────────────────────────────────────────────────────
    if not args.no_commit:
        print()
        git_commit(target_date)
    else:
        print("[INFO] Skipping git commit (--no-commit).")

    print()
    print(f"[DONE] {out_path}")


if __name__ == "__main__":
    main()
