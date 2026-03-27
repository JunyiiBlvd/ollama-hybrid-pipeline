#!/usr/bin/env python3
# log_view.py — v2
#
# Changes from v1:
#   - --review mode added: interactive queue drain for null manual_correct entries.
#     Shows each unmarked entry in sequence, prompts y/n/skip. Requires a note
#     before saving — enforces the rule that marks must include a note stating
#     what was observed. Run after any session to drain the review queue.
#     Exits cleanly on Ctrl-C.
#
# Usage: python3 log_view.py
#        python3 log_view.py --unmarked     (only show correct=null entries)
#        python3 log_view.py --fail         (only show correct=false)
#        python3 log_view.py --mark 5 true  (set entry 5 correct=true)
#        python3 log_view.py --mark 5 false "wrong model selected"
#        python3 log_view.py --review       (interactive: mark all null entries)

import json
import argparse
from pathlib import Path
from config import LOG_FILE


def load_entries():
    with open(LOG_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]

def save_entries(entries):
    with open(LOG_FILE, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

def display(entries, filter_fn=None):
    shown = 0
    for i, e in enumerate(entries):
        if filter_fn and not filter_fn(e):
            continue
        shown += 1
        correct = e.get("correct")
        status = "[ ] unmarked" if correct is None else ("[+] correct" if correct else "[-] wrong")
        constraint = e.get("constraint_injected", "none")
        print(f"\n{'─'*60}")
        print(f"  [{i:02d}] {e['ts'][:16]}  {status}")
        print(f"  Prompt:     {e['prompt'][:80]}")
        print(f"  Model:      {e['model']}  |  Skill: {e['skill_loaded']}  |  Constraint: {constraint}")
        print(f"  Tasks:      {e['task_scores']}")
        print(f"  Output:     {e['output_preview'][:120]}...")
        if e.get("notes"):
            print(f"  Notes:      {e['notes']}")
        if e.get("retried"):
            print(f"  Retried:    true")
    print(f"\n{'─'*60}")
    print(f"  Showing {shown}/{len(entries)} entries\n")

def mark(entries, index, correct, note=""):
    entries[index]["correct"] = correct
    if note:
        entries[index]["notes"] = note
    save_entries(entries)
    print(f"  Entry {index:02d} marked correct={correct}" + (f' — "{note}"' if note else ""))


def review(entries):
    """
    Interactive queue drain. Shows each null entry in sequence and prompts for
    a verdict and note. Note is required — a mark without a note is not accepted.
    Press Ctrl-C at any time to stop and save progress.
    """
    unmarked = [(i, e) for i, e in enumerate(entries) if e.get("correct") is None]
    if not unmarked:
        print("  No unmarked entries. Queue is empty.")
        return

    print(f"\n  {len(unmarked)} unmarked entries to review. Ctrl-C to stop.\n")

    reviewed = 0
    try:
        for i, e in unmarked:
            print(f"\n{'─'*60}")
            print(f"  [{i:02d}] {e['ts'][:16]}")
            print(f"  Prompt:  {e['prompt'][:100]}")
            print(f"  Model:   {e['model']}  |  Skill: {e['skill_loaded']}  |  Path: {e.get('router_path','?')}")
            print(f"  Tasks:   {e['task_scores']}")
            print(f"  Output:  {e['output_preview'][:150]}...")
            if e.get("retried"):
                print(f"  Retried: true")
            if e.get("source"):
                print(f"  Source:  {e['source']}")

            while True:
                verdict = input("\n  Correct? [y/n/s=skip] → ").strip().lower()
                if verdict == "s":
                    print("  Skipped.")
                    break
                if verdict not in ("y", "n"):
                    print("  Enter y, n, or s.")
                    continue
                # Note is required
                while True:
                    note = input("  Note (required — routing, skill, eval observations): ").strip()
                    if note:
                        break
                    print("  Note cannot be empty. State what you observed.")
                correct = verdict == "y"
                mark(entries, i, correct, note)
                reviewed += 1
                break

    except KeyboardInterrupt:
        print(f"\n\n  Stopped. {reviewed} entries marked.")
        return

    print(f"\n{'─'*60}")
    print(f"  Review complete. {reviewed}/{len(unmarked)} entries marked.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unmarked", action="store_true")
    parser.add_argument("--fail",     action="store_true")
    parser.add_argument("--review",   action="store_true")
    parser.add_argument("--mark",     nargs="+", metavar=("INDEX", "TRUE/FALSE"))
    args = parser.parse_args()

    entries = load_entries()

    if args.mark:
        idx   = int(args.mark[0])
        val   = args.mark[1].lower() == "true"
        note  = args.mark[2] if len(args.mark) > 2 else ""
        mark(entries, idx, val, note)
    elif args.review:
        review(entries)
    elif args.unmarked:
        display(entries, lambda e: e.get("correct") is None)
    elif args.fail:
        display(entries, lambda e: e.get("correct") is False)
    else:
        display(entries)
