# config.py
# v2 — Added backend config: BACKEND, BACKEND_URLS, MODEL_ALIASES
#
# Changes from v1:
# - Added BACKEND, BACKEND_URLS, MODEL_ALIASES for backend.py adapter
# - No interface changes — existing OLLAMA_URL/OLLAMA_CHAT_URL unchanged

import os
from pathlib import Path

_OLLAMA_BASE    = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_URL      = _OLLAMA_BASE + "/api/generate"
OLLAMA_CHAT_URL = _OLLAMA_BASE + "/api/chat"

_PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "qwen2.5:14b")
_FAST_MODEL    = os.getenv("FAST_MODEL", "llama3.2")

# --- Backend ---
# Options: "ollama" | "llamacpp" | "vllm"
BACKEND = os.getenv("PIPELINE_BACKEND", "ollama")

# Base URLs per backend (only the active backend is used at runtime)
BACKEND_URLS = {
    "ollama":   _OLLAMA_BASE,
    "llamacpp": os.getenv("LLAMACPP_URL", "http://127.0.0.1:8080"),
    "vllm":     os.getenv("VLLM_URL", "http://127.0.0.1:8000"),
}

# Model name translation table
# Maps internal model names to backend-specific names.
# Ollama uses Modelfile tags, llama.cpp uses file paths, vLLM uses HF identifiers.
# Pass-through: if a name is absent from the active backend's dict, it is used unchanged.
MODEL_ALIASES = {
    "ollama": {
        # Ollama native names match MODEL_MAP values — no translation needed
    },
    "llamacpp": {
        _PRIMARY_MODEL: "qwen2.5-14b-q4_k_m.gguf",   # placeholder — update to actual filename
        _FAST_MODEL:    "llama-3.2-3b-q4_k_m.gguf",  # placeholder — update to actual filename
    },
    "vllm": {
        _PRIMARY_MODEL: "Qwen/Qwen2.5-14B-Instruct",         # placeholder — update to actual HF id
        _FAST_MODEL:    "meta-llama/Llama-3.2-3B-Instruct",  # placeholder
    },
}

# Vault base path — your knowledge, skills, and memory directory
VAULT = Path(os.getenv("PIPELINE_VAULT_PATH", "./vault"))

# Log output location
LOG_FILE = VAULT / "AI/memory/routing-log.jsonl"

# Map task categories to model names
MODEL_MAP = {
    "coding":            _PRIMARY_MODEL,
    "linux_admin":       _PRIMARY_MODEL,
    "ai_dev":            _PRIMARY_MODEL,
    "domain_specialist": _PRIMARY_MODEL,
    "reasoning":         _PRIMARY_MODEL,
    "session_memory":    _PRIMARY_MODEL,
    "fast":              _FAST_MODEL,
    "default":           _PRIMARY_MODEL,
}

# Skill files to inject per dominant task
SKILL_FILES = {
    "domain_specialist": VAULT / "AI/skills/domain-specialist.md",
    "coding":            VAULT / "AI/skills/coding.md",
    "linux_admin":       VAULT / "AI/skills/linux-security.md",
    "ai_dev":            VAULT / "AI/skills/ai-development.md",
    "reasoning":         VAULT / "AI/skills/base-context.md",
    "session_memory":    VAULT / "AI/skills/session-memory.md",
    "fast":              VAULT / "AI/skills/fast.md",
}

# Base context always injected regardless of task
BASE_CONTEXT_FILE = VAULT / "AI/skills/base-context.md"

# Max characters of skill content to inject
# Keeps system prompt reasonable, avoids context overflow
SKILL_INJECT_LIMIT   = 3000
BASE_INJECT_LIMIT    = 1000
CONTEXT_INJECT_LIMIT = 4000  # context-loader budget — separate from skill and base limits

# Disambiguation — terms that collide with common meanings
# When detected, inject clarification before the prompt
DISAMBIGUATION_MAP = {
    "pipeline": {
        "context_clues": ["route", "skill", "constraint", "router", "ci/cd", "data pipeline"],
        "clarification": (
            "Note: 'pipeline' refers to this local AI pipeline system, not "
            "a CI/CD pipeline or data engineering pipeline."
        )
    },
    "agent": {
        "context_clues": ["ollama", "skill", "route", "pipeline"],
        "clarification": (
            "Note: 'agent' refers to a local LLM agent "
            "running in the local AI pipeline. "
            "It is NOT a cloud AI agent service."
        )
    },
}

# Task scoring signals
TASK_SIGNALS = {
    "coding": [
        "script", "code", "function", "python", "bash",
        "automate", "build", "implement", "write a program",
        "class", "method", "loop", "debug", "fix this",
        "error", "traceback", "import", "module", "write"
    ],
    "linux_admin": [
        "docker", "container", "systemd", "journalctl",
        "permissions", "chmod", "ufw", "firewall", "apt",
        "service", "daemon", "port", "mount", "fstab",
        "logs", "deployment", "nginx", "ssh", "sudo",
        "cron", "crontab", "iptables", "netstat", "process"
    ],
    "ai_dev": [
        "ollama", "ollama api", "model", "vram", "gpu", "open-webui",
        "knowledge base", "rag", "embedding", "inference",
        "modelfile", "context window", "temperature", "tokens"
    ],
    "domain_specialist": [
        # Add your domain-specific keywords here.
        # Example: a FastAPI backend project might use:
        # "fastapi", "endpoint", "router", "middleware", "websocket",
        # "pydantic", "schema", "dependency", "lifespan",
    ],
    "reasoning": [
        "analyze", "compare", "evaluate", "why", "optimize",
        "design", "architecture", "should i", "should i use", "what is",
        "how does", "best practice", "tradeoff", "tradeoffs", "recommend",
        "understand", "review", "assess", "difference between",
        "explain", "walk me through", "help me understand",
        "describe", "overview", "instead", "rather than",
        "pros and cons", "when to use", "which approach", "versus"
    ],
    "session_memory": [
        "session memory", "session-memory", "session summary",
        "generate memory", "memory file", "routing-log",
        "summarize session", "summarise session", "daily summary",
        "what did we do", "what was done", "session recap",
        "write a summary", "memory for today", "memory for this session",
        "log entries", "session log", "pipeline log"
    ],
    "fast": [
        "what time", "quick question", "yes or no",
        "define", "what does", "what flag", "what port",
        "by default", "spelling", "translate"
    ]
}

# Capability weights per task
# TODO: replace with data-backed weights after 15+ routing log entries
MODEL_CAPABILITIES = {
    _PRIMARY_MODEL: {
        "coding":            3,
        "linux_admin":       3,
        "ai_dev":            4,
        "domain_specialist": 4,
        "reasoning":         4,
        "session_memory":    4,
        "fast":              1,
    },
    _FAST_MODEL: {
        "coding":            1,
        "linux_admin":       1,
        "ai_dev":            1,
        "domain_specialist": 0,
        "reasoning":         1,
        "session_memory":    0,
        "fast":              4,
    },
}

# Priority order for tie-breaking in skill selection
# Higher priority = wins when scores are equal
TASK_PRIORITY = ["domain_specialist", "linux_admin", "ai_dev", "coding", "session_memory", "reasoning", "fast"]
