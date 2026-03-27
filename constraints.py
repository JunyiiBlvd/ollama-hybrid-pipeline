import os

# constraints.py
# Short, surgical constraint blocks injected AFTER skill content.
# These are guaranteed to reach the model — never truncated.
# Keep each block SHORT. Rules only, no prose explanation.
# This is NOT a skill file — it is an enforcement layer.

CONSTRAINTS = {
    "domain_specialist": """
## HARD CONSTRAINTS — VIOLATIONS WILL BE DETECTED AND FLAGGED

# Example domain_specialist constraints:
# FRAMEWORK: Use [YourFramework] only — never create new instances directly
# PARSING: json.loads() only — eval() is forbidden
# PATHS: Use configured base paths — never hardcode absolute paths
""",

    "coding": f"""
## HARD CONSTRAINTS

PARSING: json.loads() only — eval() is forbidden
PATHS: Use {os.getenv("PIPELINE_VAULT_PATH", "./vault")} not hardcoded absolute paths
ERRORS: All external calls must have try/except
""",

    "linux_admin": f"""
## HARD CONSTRAINTS

VAULT: local file vault at {os.getenv("PIPELINE_VAULT_PATH", "./vault")} (NOT HashiCorp Vault)
VAULT BACKUP: git add . && git commit && git push (not tar/rsync)
STORAGE: base path is {os.getenv("PIPELINE_STORAGE_PATH", "./storage")} not /var/
""",

    "ai_dev": """
## HARD CONSTRAINTS

OLLAMA API: /api/generate for single-shot calls — /api/chat for threaded conversation (thread_store.py)
CONTAINER: open-webui, network mode host
""",
}


def load_constraints(task_type: str) -> str:
    """
    Return constraint block for the given task type.
    Returns empty string if no constraints defined — never crashes.
    """
    return CONSTRAINTS.get(task_type, "")
