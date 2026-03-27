#!/usr/bin/env python3
# tools.py — Tool registry for the local AI pipeline orchestrator
# v4
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
#     → Writes content to a file in VAULT. Creates parent dirs. Blocks path traversal.
#       Returns the absolute path written.
#
# Tool registry:
#   TOOL_REGISTRY — dict mapping tool name → callable
#   call_tool(name, args) — looks up and calls a tool by name

from pathlib import Path
from config import VAULT

_FILE_READ_CAP = 4000


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
    Write content to a file in VAULT. Creates parent directories.
    relative_path is relative to the vault root.
    Example: "AI/memory/orchestrator-output.md"

    Returns the absolute path of the file written, or an error string.
    """
    result = _safe_vault_path(relative_path)
    if isinstance(result, str):
        return result
    path = result

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)
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
