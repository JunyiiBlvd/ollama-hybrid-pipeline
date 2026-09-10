#!/usr/bin/env python3
# tools.py — Tool registry for the pipeline orchestrator
# v5
#
# Changes from v4:
#   - SECURITY (3.1): write_vault_file now caps content at _FILE_WRITE_CAP
#     (50 000 bytes). Content above the cap is truncated and a
#     "[truncated at 50000 bytes]" marker is appended. The return value
#     includes a WARNING suffix when truncation fires. Previously the tool
#     accepted whatever the model produced — a step whose prompt said "dump
#     the full context" could write megabytes to the vault.
#   - SECURITY (3.3): write_vault_file now enforces an allowlist on the
#     resolved path. Only AI/memory/ and AI/notes/ are valid write targets.
#     All other paths are rejected with a clear error. The check is on the
#     resolved Path object (not the raw string) so AI/memory/../foo bypasses
#     are not possible. Previously _safe_vault_path only blocked escape from
#     the vault root — a model could overwrite Projects/pipeline-setup/
#     router/tools.py or CLAUDE.md.
#
# Changes from v3:
#   - web_search() removed entirely. DDG scraping was blocked by bot detection;
#     Brave Search API (the replacement) requires sending queries to an external
#     server, exposing the content of orchestrator prompts and creating an
#     indirect prompt injection surface via web results. Decision: pipeline
#     steps already have access to the local knowledge base, RAG index, and
#     memory — no external search needed for the current use case.
#   - _load_creds(), _get_brave_key(), and all Brave/DDG imports removed.
#   - CLI updated to reflect remaining tools only.
#
# Available tools:
#   read_vault_file(relative_path)
#     → Reads a file from VAULT. relative_path is relative to VAULT root.
#       Blocks any path that resolves outside VAULT. Caps at 4000 chars.
#
#   write_vault_file(relative_path, content)
#     → Writes content to a file in VAULT under AI/memory/ or AI/notes/ only.
#       Blocks path traversal and paths outside the allowlist. Caps at 50 000
#       bytes. Returns the absolute path written, with a WARNING suffix if
#       content was truncated.
#
# Tool registry:
#   TOOL_REGISTRY — dict mapping tool name → callable
#   call_tool(name, args) — looks up and calls a tool by name

from pathlib import Path
from config import VAULT

_FILE_READ_CAP  = 4_000
_FILE_WRITE_CAP = 50_000

# Resolved allowlist for write_vault_file — checked against the resolved Path,
# not the raw string, so AI/memory/../foo traversals are caught.
_VAULT_RESOLVED = VAULT.resolve()
_WRITE_ALLOWLIST = [
    str(_VAULT_RESOLVED / "AI" / "memory") + "/",
    str(_VAULT_RESOLVED / "AI" / "notes") + "/",
]


# ─── Vault File Tools ─────────────────────────────────────────────────────────

def _safe_vault_path(relative_path: str) -> Path | str:
    """
    Resolve relative_path within VAULT. Returns the resolved Path if safe,
    or an error string if the resolved path escapes VAULT.
    """
    try:
        target = (VAULT / relative_path).resolve()
        vault_resolved = VAULT.resolve()
        if not str(target).startswith(str(vault_resolved) + "/") and target != vault_resolved:
            return f"[vault] ERROR: path escapes vault root: {relative_path!r}"
        return target
    except Exception as e:
        return f"[vault] ERROR resolving path: {e}"


def read_vault_file(relative_path: str) -> str:
    """
    Read a file from VAULT. relative_path is relative to the vault root.
    Example: "AI/memory/2026-03-26-session.md"

    Content is capped at _FILE_READ_CAP chars. Returns error string if the
    file does not exist or the path escapes the vault.
    """
    result = _safe_vault_path(relative_path)
    if isinstance(result, str):
        return result
    path = result

    if not path.exists():
        return f"[read_vault_file] ERROR: file not found: {relative_path!r}"
    if not path.is_file():
        return f"[read_vault_file] ERROR: not a file: {relative_path!r}"

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        if len(content) > _FILE_READ_CAP:
            content = content[:_FILE_READ_CAP] + f"\n[truncated — {len(content)} chars total]"
        return content
    except Exception as e:
        return f"[read_vault_file] ERROR reading {relative_path!r}: {e}"


def write_vault_file(relative_path: str, content: str) -> str:
    """
    Write content to a file in VAULT under AI/memory/ or AI/notes/ only.
    relative_path is relative to the vault root.
    Example: "AI/memory/orchestrator-output.md"

    Validation chain:
      1. _safe_vault_path — rejects any path that resolves outside VAULT.
      2. Allowlist check — rejects resolved paths outside AI/memory/ or AI/notes/.
         Checked on the resolved Path so AI/memory/../foo traversals are caught.
      3. Size cap — content above _FILE_WRITE_CAP (50 000 bytes) is truncated
         and a marker is appended.

    Returns the absolute path written (with a WARNING suffix if truncated),
    or an error string on rejection.
    """
    result = _safe_vault_path(relative_path)
    if isinstance(result, str):
        return result
    path = result

    # Allowlist: reject any resolved path outside AI/memory/ or AI/notes/
    if not any(str(path).startswith(prefix) for prefix in _WRITE_ALLOWLIST):
        allowed = " or ".join(f"AI/{p.split('/AI/')[1].rstrip('/')}" for p in _WRITE_ALLOWLIST)
        return (
            f"[write_vault_file] ERROR: writes restricted to {allowed}: {relative_path!r}"
        )

    # Size cap: truncate and mark rather than silently writing huge files
    warning = ""
    if len(content) > _FILE_WRITE_CAP:
        content = content[:_FILE_WRITE_CAP] + "\n[truncated at 50000 bytes]"
        warning = f" [WARNING: content exceeded {_FILE_WRITE_CAP} bytes and was truncated]"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path) + warning
    except Exception as e:
        return f"[write_vault_file] ERROR writing {relative_path!r}: {e}"


# ─── Tool Registry ────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, callable] = {
    "read_vault_file":  read_vault_file,
    "write_vault_file": write_vault_file,
}


def call_tool(name: str, args: dict) -> str:
    """
    Call a registered tool by name with keyword arguments.
    Returns the tool's string output, or an error string for unknown tools.
    """
    if name not in TOOL_REGISTRY:
        available = ", ".join(TOOL_REGISTRY.keys())
        return f"[tool] ERROR: unknown tool: {name!r}. Available: {available}"
    try:
        return TOOL_REGISTRY[name](**args)
    except TypeError as e:
        return f"[tool] ERROR calling {name!r} with args {args}: {e}"


# ─── CLI (diagnostic) ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 tools.py read_vault_file 'AI/memory/2026-03-26-session.md'")
        print("  python3 tools.py write_vault_file 'AI/memory/test.md' 'content here'")
        sys.exit(1)

    tool_name = sys.argv[1]
    if tool_name == "read_vault_file":
        print(read_vault_file(sys.argv[2]))
    elif tool_name == "write_vault_file" and len(sys.argv) >= 4:
        print(write_vault_file(sys.argv[2], sys.argv[3]))
    else:
        print(f"[tools] Unknown command: {sys.argv[1:]}")
        sys.exit(1)
