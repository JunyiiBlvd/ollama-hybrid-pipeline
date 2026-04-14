#!/usr/bin/env python3
# pshell.py — Persistent interactive shell for the local AI pipeline
# v1.3
#
# Changes from v1.2:
#   - requests.post() timeout changed to (10, None) — fixes read timeout on slow model generation
#   - stream interrupt handler logs exception type to stderr for diagnostics
#
# Changes from v1.1:
#   - fetch_response() iter_lines() loop wrapped in inner try/except for
#     ChunkedEncodingError and ConnectionError; mid-stream interruptions now
#     return buffered content + "[stream interrupted — response may be incomplete]"
#     instead of discarding everything; main loop continues unaffected
#
# Changes from v1.0:
#   - SCRIPT_DIR = Path(__file__).resolve().parent anchors all paths to script location;
#     pshell can now be launched from any working directory
#   - /models queries Ollama /api/tags live; cross-references MODEL_MAP;
#     shows [pulled]/[NOT PULLED] status per registered model;
#     shows unregistered models available in Ollama
#
# What v1.0 includes:
#   - REPL with readline history; connects to pipeline_api.py via SSE streaming
#   - Routing metadata footer stripped and shown as header before response body
#   - Conversation history capped at MAX_TURNS=20; oldest turn dropped with notice
#   - Slash commands: /exit /quit /clear /switch /log /memory /models /help
#   - /switch validates port, re-runs preflight
#   - /log: last 5 entries from active pipeline's routing-log.jsonl
#   - /memory: 3 most recent .md files in vault memory directory by mtime
#   - /models: MODEL_MAP loaded from active pipeline's config.py at runtime
#   - Session memory written on clean exit if session had ≥3 turns
#   - Config loaded via importlib.util (no sys.path mutation, no module conflicts)

import sys
import os
import re
import json
import readline
import requests
import importlib.util
from datetime import datetime
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
DEFAULT_PORT = int(os.getenv("PIPELINE_PORT", "11436"))
MAX_TURNS    = 20
MODEL_ID     = "local-pipeline"

VAULT = Path(os.getenv("PIPELINE_VAULT_PATH", "./vault"))


def _load_pipeline_config(port: int):
    """Load config.py from the same directory as pshell.py."""
    path = SCRIPT_DIR / "config.py"
    spec = importlib.util.spec_from_file_location("_pipeline_config", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def preflight(port: int) -> bool:
    try:
        r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def fetch_response(messages: list, port: int) -> tuple:
    """
    POST to /v1/chat/completions (stream=True), buffer full SSE stream.
    Splits on footer marker appended by pipeline_api.py:
        \\n\\n---\\n*routed → {model} | skill: {skill} | path: {path}*
    Returns: (body: str, routing: dict)
    """
    url     = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {"model": MODEL_ID, "messages": messages, "stream": True}
    tokens: list = []

    try:
        with requests.post(url, json=payload, stream=True, timeout=(10, None)) as resp:
            resp.raise_for_status()
            try:
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw if isinstance(raw, str) else raw.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        token = json.loads(data)["choices"][0]["delta"].get("content", "")
                        if token:
                            tokens.append(token)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError) as e:
                print(f"[debug] stream ended: {type(e).__name__}", file=sys.stderr)
                partial = _strip_ansi("".join(tokens).strip())
                notice  = "\n\n[stream interrupted — response may be incomplete]"
                return (partial + notice) if partial else notice, {}
    except requests.exceptions.ConnectionError:
        return "[ERROR] Pipeline not reachable. Is it running?", {}
    except requests.exceptions.Timeout:
        return "[ERROR] Pipeline connect timeout (10s).", {}
    except requests.exceptions.HTTPError as e:
        return f"[ERROR] HTTP {e}", {}

    full = "".join(tokens)
    parts = full.split("\n\n---\n", 1)
    body  = parts[0].strip()
    routing = {}

    if len(parts) == 2:
        m = re.search(r'\*routed → (.+?) \| skill: (.+?) \| path: (.+?)\*', parts[1])
        if m:
            routing = {"model": m.group(1).strip(),
                       "skill": m.group(2).strip(),
                       "path":  m.group(3).strip()}
    return _strip_ansi(body), routing


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mGKHFJA-Z]')

def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub('', text)


def display_header(routing: dict) -> None:
    print(f"\n[{routing.get('model','?')} | skill:{routing.get('skill','?')} | {routing.get('path','?')}]")


def cmd_log(port: int) -> None:
    try:
        log_file = Path(_load_pipeline_config(port).LOG_FILE)
    except Exception as e:
        print(f"[log] config load failed: {e}"); return

    if not log_file.exists():
        print(f"[log] file not found: {log_file}"); return

    try:
        recent = [l for l in log_file.read_text().splitlines() if l.strip()][-5:]
        if not recent:
            print("[log] no entries"); return
        for line in recent:
            try:
                e = json.loads(line)
                print(f"  {e.get('ts','?')[:19]} | {e.get('model','?')} | "
                      f"skill:{e.get('skill_loaded','?')} | {e.get('router_path','?')} | "
                      f"{e.get('prompt','')[:60]!r}")
            except json.JSONDecodeError:
                print("  [malformed entry]")
    except Exception as e:
        print(f"[log] read failed: {e}")


def cmd_memory() -> None:
    mem_dir = VAULT / "AI/memory"
    if not mem_dir.exists():
        print("[memory] directory not found"); return
    files = sorted(mem_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)[:3]
    if not files:
        print("[memory] no .md files found"); return
    for f in files:
        print(f"  {f.name}")


def cmd_models(port: int) -> None:
    import urllib.request as _urllib_req

    try:
        cfg = _load_pipeline_config(port)
    except Exception as e:
        print(f"[models] config load failed: {e}"); return

    try:
        with _urllib_req.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            ollama_live = {m["name"] for m in json.loads(r.read()).get("models", [])}
    except Exception:
        ollama_live = set()

    # Normalize: treat "model" and "model:latest" as the same for matching
    def _normalize(tag: str) -> str:
        return tag[:-7] if tag.endswith(":latest") else tag

    ollama_normalized = {_normalize(t) for t in ollama_live}
    model_map = cfg.MODEL_MAP

    print(f"\n[models] active pipeline → port {port}\n")
    print("  REGISTERED (MODEL_MAP):")
    for alias, tag in model_map.items():
        status = "[pulled]" if _normalize(tag) in ollama_normalized else "[NOT PULLED]"
        print(f"    {alias:<20} → {tag}  {status}")

    registered_normalized = {_normalize(t) for t in model_map.values()}
    unregistered = {t for t in ollama_live if _normalize(t) not in registered_normalized}
    if unregistered:
        print("\n  AVAILABLE IN OLLAMA (not in pipeline):")
        for tag in sorted(unregistered):
            print(f"    {tag}")
    print()


def cmd_help() -> None:
    print("  /exit, /quit   exit (saves session if ≥3 turns)\n"
          "  /clear         reset conversation context\n"
          "  /switch <port> switch pipeline port (default: 11436)\n"
          "  /log           last 5 routing log entries for active pipeline\n"
          "  /memory        3 most recent session memory files\n"
          "  /models        MODEL_MAP for active pipeline\n"
          "  /help          this help text")


def save_session_memory(session_id: str, port: int, model: str, prompts: list) -> None:
    mem_dir = VAULT / "AI/memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    fname = mem_dir / f"{datetime.now().strftime('%Y-%m-%d')}-pshell-{session_id[:8]}.md"
    fname.write_text("\n".join([
        f"# pshell session — {session_id}",
        f"port: {port}  |  model: {model}  |  turns: {len(prompts)}",
        "", "## Prompts",
        *[f"- {p}" for p in prompts], "",
    ]))
    print(f"[session saved → {fname.name}]")


def main() -> None:
    port = DEFAULT_PORT
    if not preflight(port):
        print(f"[ERROR] Pipeline not reachable on port {port}")
        print(f"  → start: uvicorn pipeline_api:app --port {port}")
        sys.exit(1)

    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    messages:   list = []
    prompts:    list = []
    last_model: str  = MODEL_ID

    readline.parse_and_bind("tab: complete")
    print(f"pipeline shell | {last_model} | port {port}")
    print("type /help for commands, Ctrl+D to exit\n")

    while True:
        try:
            user_input = input("pipeline ▸ ").strip()
        except KeyboardInterrupt:
            print(); continue
        except EOFError:
            print("\nsession ended.")
            if len(prompts) >= 3:
                save_session_memory(session_id, port, last_model, prompts)
            sys.exit(0)

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd   = parts[0].lower()
            arg   = parts[1] if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit"):
                print("session ended.")
                if len(prompts) >= 3:
                    save_session_memory(session_id, port, last_model, prompts)
                sys.exit(0)
            elif cmd == "/clear":
                messages = []; print("context cleared.")
            elif cmd == "/switch":
                if not arg:
                    print("usage: /switch <port>")
                else:
                    try:
                        new_port = int(arg.strip())
                    except ValueError:
                        print(f"[switch] invalid port: {arg!r}"); continue
                    if not (1024 <= new_port <= 65535):
                        print(f"[switch] rejected — port must be 1024–65535")
                    elif not preflight(new_port):
                        print(f"[switch] port {new_port} not reachable — pipeline running?")
                    else:
                        port = new_port
                        print(f"pipeline shell | {MODEL_ID} | port {port}")
            elif cmd == "/log":     cmd_log(port)
            elif cmd == "/memory":  cmd_memory()
            elif cmd == "/models":  cmd_models(port)
            elif cmd == "/help":    cmd_help()
            else:
                print(f"unknown command: {cmd}  (try /help)")
            continue

        messages.append({"role": "user", "content": user_input})
        prompts.append(user_input)

        body, routing = fetch_response(messages, port)

        if routing:
            display_header(routing)
            last_model = routing.get("model", last_model)

        print(body)
        print()

        messages.append({"role": "assistant", "content": body})

        if len(messages) > MAX_TURNS * 2:
            messages = messages[2:]
            print("[context trimmed — oldest turn removed]")


if __name__ == "__main__":
    main()
