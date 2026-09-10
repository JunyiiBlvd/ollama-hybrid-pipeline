#!/usr/bin/env python3
# thread_store.py — Conversation thread persistence for the pipeline pipeline
# v3 — Wired close_thread() Ollama call to backend.py adapter
#
# Changes from v2:
# - Replaced direct Ollama POST in close_thread() with call_backend() from backend.py
# - No interface changes — callers unaffected
#
# Changes from v1 (now v2):
#   - close_thread(thread_id) added — calls the primary model via /api/generate to
#     summarize a thread (what was decided, built, and remains open), writes the
#     summary to AI/memory/THREAD_ID-closed.md with YAML frontmatter, moves the
#     thread JSON to threads/archive/ (no permanent deletion), then calls
#     rag_index.rebuild_memory() so the summary is immediately queryable by
#     context_loader without a manual --rebuild. Returns the path of the summary.
#
# What this does:
#   Saves and loads conversation history so the model has full context of what
#   was said in the current thread. Each thread is a JSON file in AI/memory/threads/.
#   The system message (skill + context + constraints) is NOT stored — it is
#   regenerated fresh on every call so it always reflects the current state of
#   skill files and context.
#
# Thread file format:
#   AI/memory/threads/THREAD_ID.json
#   {
#     "thread_id": "20260326T143021",
#     "created":   "2026-03-26T14:30:21",
#     "messages":  [
#       {"role": "user",      "content": "..."},
#       {"role": "assistant", "content": "..."}
#     ]
#   }
#
# Usage from run_task.py:
#   thread = load_thread(thread_id)          # load existing thread
#   thread = new_thread()                    # start a new thread
#   thread = append_exchange(thread, prompt, response)
#   save_thread(thread)
#   list_threads()                           # print recent threads
#   close_thread(thread_id)                  # summarize and archive a thread

import os
import json
import rag_index
from datetime import datetime
from pathlib import Path
from config import VAULT
from backend import call_backend, BackendConnectionError

THREADS_DIR  = VAULT / "AI/memory/threads"
MEMORY_DIR   = VAULT / "AI/memory"
ARCHIVE_DIR  = THREADS_DIR / "archive"

_SUMMARIZE_MODEL  = os.getenv("PRIMARY_MODEL", "qwen2.5:14b")
_SUMMARIZE_PROMPT = """\
You are summarizing a completed conversation thread for long-term memory.
Read the conversation below and write a concise summary under 500 words.
Cover three things:
1. What was decided
2. What was built or changed
3. What remains open or unresolved

Write in plain prose. No code fences. No bullet lists longer than 5 items.
No headings beyond the three above. Be specific — name files, functions, and
commands where relevant. Omit filler and meta-commentary.

--- CONVERSATION ---
{conversation}
--- END CONVERSATION ---
"""


# ─── Thread Lifecycle ─────────────────────────────────────────────────────────

def new_thread() -> dict:
    """Create a new empty thread. Does not save to disk."""
    thread_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    return {
        "thread_id": thread_id,
        "created":   datetime.now().isoformat(),
        "messages":  [],
    }


def load_thread(thread_id: str) -> dict | None:
    """
    Load an existing thread by ID.
    Returns None if the thread file does not exist.
    """
    path = THREADS_DIR / f"{thread_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_thread(thread: dict) -> Path:
    """Write thread to disk. Creates THREADS_DIR if needed."""
    THREADS_DIR.mkdir(parents=True, exist_ok=True)
    path = THREADS_DIR / f"{thread['thread_id']}.json"
    path.write_text(json.dumps(thread, indent=2))
    return path


def append_exchange(thread: dict, user_prompt: str, assistant_response: str) -> dict:
    """Append a user/assistant exchange to the thread. Returns updated thread."""
    thread["messages"].append({"role": "user",      "content": user_prompt})
    thread["messages"].append({"role": "assistant", "content": assistant_response})
    return thread


# ─── Thread Listing ───────────────────────────────────────────────────────────

def list_threads(n: int = 10) -> list[dict]:
    """
    Return the N most recently modified thread files as summary dicts.
    Each entry: {"thread_id", "created", "turns", "preview"}
    """
    if not THREADS_DIR.exists():
        return []

    files = sorted(
        THREADS_DIR.glob("*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )[:n]

    summaries = []
    for f in files:
        try:
            t = json.loads(f.read_text())
            messages = t.get("messages", [])
            turns = len(messages) // 2
            first_user = next(
                (m["content"][:80] for m in messages if m["role"] == "user"),
                "(empty)"
            )
            summaries.append({
                "thread_id": t.get("thread_id", f.stem),
                "created":   t.get("created", "")[:16],
                "turns":     turns,
                "preview":   first_user,
            })
        except Exception:
            continue

    return summaries


def print_threads(n: int = 10) -> None:
    """Print a formatted list of recent threads to stdout."""
    threads = list_threads(n)
    if not threads:
        print("No threads found.")
        return

    print(f"{'Thread ID':<20}  {'Created':<16}  {'Turns':>5}  Preview")
    print("─" * 80)
    for t in threads:
        print(f"{t['thread_id']:<20}  {t['created']:<16}  {t['turns']:>5}  {t['preview']}")


# ─── Thread Summary (for memory injection) ────────────────────────────────────

def get_thread_as_messages(thread: dict) -> list[dict]:
    """Return a copy of the messages list for passing to /api/chat."""
    return list(thread.get("messages", []))


def summarize_thread_for_memory(thread: dict) -> str:
    """
    Return a compact text representation of the thread for use as a memory
    file or for injection into context_loader. Truncates each exchange to
    keep the summary under 2000 chars total.
    """
    messages = thread.get("messages", [])
    lines = [f"# Thread {thread['thread_id']} — {thread.get('created', '')[:10]}\n"]

    pairs = zip(messages[::2], messages[1::2])
    for i, (user_msg, asst_msg) in enumerate(pairs, 1):
        u = user_msg["content"][:200]
        a = asst_msg["content"][:300]
        lines.append(f"**Turn {i}**\nUser: {u}\nAssistant: {a}\n")

    return "\n".join(lines)


# ─── Thread Close / Archive ───────────────────────────────────────────────────

def close_thread(thread_id: str) -> Path:
    """
    Summarize a thread and archive it.

    Steps:
      1. Load the thread JSON from THREADS_DIR/THREAD_ID.json.
      2. Build a full conversation transcript.
      3. Call the primary model via /api/generate to produce the summary.
      4. Write summary to MEMORY_DIR/THREAD_ID-closed.md with YAML frontmatter.
      5. Move the source JSON to ARCHIVE_DIR/THREAD_ID.json.
      6. Return the path of the written summary file.

    Raises:
      FileNotFoundError  — thread JSON does not exist.
      RuntimeError       — Ollama call fails or returns empty.
    """
    thread_path = THREADS_DIR / f"{thread_id}.json"
    if not thread_path.exists():
        raise FileNotFoundError(f"Thread not found: {thread_path}")

    thread = json.loads(thread_path.read_text())
    messages = thread.get("messages", [])

    # Build plain-text transcript for the summarizer
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "").strip()
        lines.append(f"[{role}]\n{content}")
    conversation = "\n\n".join(lines)

    prompt = _SUMMARIZE_PROMPT.format(conversation=conversation)

    # Call backend
    try:
        summary = call_backend(
            _SUMMARIZE_MODEL, prompt, stream=False, temperature=0.3
        )
    except (BackendConnectionError, Exception) as e:
        raise RuntimeError(f"Backend call failed during close_thread: {e}") from e

    if not summary:
        raise RuntimeError("Ollama returned an empty summary.")

    # Write summary with frontmatter
    today = datetime.now().strftime("%Y-%m-%d")
    frontmatter = (
        f"---\n"
        f"date: {today}\n"
        f"thread_id: {thread_id}\n"
        f"type: thread-summary\n"
        f"---\n\n"
    )
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MEMORY_DIR / f"{thread_id}-closed.md"
    out_path.write_text(frontmatter + summary + "\n")

    # Archive the source thread JSON
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = ARCHIVE_DIR / f"{thread_id}.json"
    thread_path.rename(archive_path)

    # Rebuild the memory RAG index so the new summary is immediately queryable
    rag_index.rebuild_memory()

    return out_path


# ─── CLI (diagnostic) ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        print_threads()
    elif len(sys.argv) > 1:
        t = load_thread(sys.argv[1])
        if t:
            print(json.dumps(t, indent=2))
        else:
            print(f"Thread not found: {sys.argv[1]}")
    else:
        print("Usage: python3 thread_store.py list")
        print("       python3 thread_store.py THREAD_ID")
