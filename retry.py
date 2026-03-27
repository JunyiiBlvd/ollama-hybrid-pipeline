# retry.py — specific corrective instructions per failure type
# STUB: returns decision and corrections, does NOT execute retry

from evaluator import EvalResult

# Maps failure issue substrings to specific corrective instructions.
# Add new mappings as new failure types emerge from logs.
CORRECTION_MAP = {
    "Flask":
        "Use FastAPI only. Import existing app from api.py. "
        "Do not create a new FastAPI() instance.",

    "eval()":
        "Replace eval() with json.loads(). "
        "Example: `data = json.loads(raw_text)` not `data = eval(raw_text)`",

    "New FastAPI app":
        "Do not write `app = FastAPI()`. "
        "Import the existing app from api.py.",

    "HashiCorp":
        "This pipeline uses a local file vault, not HashiCorp Vault. "
        "Do not generate HashiCorp Vault commands.",

    # Add your domain-specific correction mappings here.
    # Format: "substring from failure issue": "corrective instruction",
}


def get_correction(failure_issue: str) -> str:
    """
    Map a failure issue description to a specific corrective instruction.
    Falls back to generic message if no mapping found — and flags it.
    """
    for keyword, correction in CORRECTION_MAP.items():
        if keyword.lower() in failure_issue.lower():
            return correction

    # Unmapped failure — return generic but flag for mapping
    print(f"[RETRY] Unmapped failure type: '{failure_issue}' "
          f"— add to CORRECTION_MAP in retry.py")
    return f"Fix this violation: {failure_issue}"


def should_retry(eval_result: EvalResult, attempt: int = 1) -> dict:
    """
    Decide whether retry is warranted.
    Returns decision dict with SPECIFIC corrections per failure.
    Does NOT call any model.
    """
    MAX_ATTEMPTS = 3

    if attempt >= MAX_ATTEMPTS:
        return {
            "retry": False,
            "reason": f"Max attempts ({MAX_ATTEMPTS}) reached",
            "corrections": []
        }

    if eval_result.passed:
        return {
            "retry": False,
            "reason": "Evaluation passed — no critical violations",
            "corrections": []
        }

    critical = [f for f in eval_result.failures if f.severity == "critical"]

    if not critical:
        return {
            "retry": False,
            "reason": "Warnings only — not worth retry cost",
            "corrections": [get_correction(f.issue) for f in eval_result.failures]
        }

    corrections = [get_correction(f.issue) for f in critical]

    return {
        "retry": True,
        "reason": f"{len(critical)} critical violation(s) detected",
        "corrections": corrections
    }


def build_retry_prompt(original_prompt: str, retry_decision: dict) -> str:
    """
    Build improved prompt for retry with specific corrections.
    NOT called automatically — foundation for future retry loop.
    """
    if not retry_decision["corrections"]:
        return original_prompt

    corrections_block = "\n".join(
        f"- {c}" for c in retry_decision["corrections"]
    )

    return (
        f"{original_prompt}\n\n"
        f"PREVIOUS ATTEMPT FAILED — APPLY THESE SPECIFIC FIXES:\n"
        f"{corrections_block}"
    )
