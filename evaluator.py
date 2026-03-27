# evaluator.py
# Version: 1
# Checks model output against known constraint rules.
# Returns structured pass/fail — does NOT call any model.
# Place in same directory as run_task.py
#
# Changes from v0:
#   - Fixed bug in evaluate(): Failure(type=...) was incorrectly set to
#     `severity` instead of `ftype`. Now correctly uses `ftype`.

from dataclasses import dataclass, field


@dataclass
class Failure:
    type: str    # "safety" | "architecture" | "integration" | "convention"
    issue: str   # human-readable description
    severity: str  # "critical" | "warning"


@dataclass
class EvalResult:
    passed: bool
    failures: list[Failure] = field(default_factory=list)
    score: float = 0.0  # 0.0 to 1.0

    def summary(self) -> str:
        if self.passed:
            return f"PASS (score: {self.score:.2f})"
        critical = [f for f in self.failures if f.severity == "critical"]
        warnings = [f for f in self.failures if f.severity == "warning"]
        return (
            f"FAIL (score: {self.score:.2f}) — "
            f"{len(critical)} critical, {len(warnings)} warnings"
        )


# ─── Rule Definitions ─────────────────────────────────────────────────────────

# Each rule is a tuple: (check_fn, failure_type, issue_text, severity)
# check_fn returns True if the VIOLATION is detected (not if it's fine)

GENERAL_RULES = [
    (
        lambda r: "eval(" in r,
        "safety",
        "eval() used for parsing — use json.loads() instead",
        "critical"
    ),
    (
        lambda r: "import os" in r and "system(" in r,
        "safety",
        "os.system() detected — use subprocess instead",
        "critical"
    ),
]

# Domain-specialist rules — customize these for your specific project.
# These rules enforce architecture and integration constraints for domain_specialist tasks.
# Example rules for a FastAPI backend domain:
DOMAIN_SPECIALIST_RULES = [
    (
        lambda r: "from flask" in r.lower() or "import flask" in r.lower(),
        "architecture",
        "Flask used — project uses FastAPI",
        "critical"
    ),
    (
        lambda r: "app = FastAPI()" in r,
        "architecture",
        "New FastAPI app created — import from existing api.py instead",
        "critical"
    ),
    # Add your project-specific rules here. Examples:
    # Check that required fields are present in responses
    # Check that specific classes/modules are used correctly
    # Check that deprecated patterns are not used
]

VAULT_RULES = [
    (
        lambda r: "hashicorp" in r.lower() or "vault status" in r.lower()
                  or "vault token" in r.lower(),
        "convention",
        "HashiCorp Vault referenced — confirm intended: this pipeline uses a local file vault",
        "critical"
    ),
]

# ─── Rule Sets Per Task Type ──────────────────────────────────────────────────

RULE_SETS = {
    "domain_specialist": GENERAL_RULES + DOMAIN_SPECIALIST_RULES,
    "coding":            GENERAL_RULES,
    "linux_admin":       GENERAL_RULES + VAULT_RULES,
    "default":           GENERAL_RULES,
}


# ─── Evaluator Function ───────────────────────────────────────────────────────

def evaluate(response: str, task_type: str = "default") -> EvalResult:
    """
    Run all applicable rules against the model response.
    Returns structured EvalResult with pass/fail and failure list.

    Does NOT call any model. Pure string analysis.
    """
    rules = RULE_SETS.get(task_type, RULE_SETS["default"])
    failures = []

    for check_fn, ftype, issue, severity in rules:
        try:
            if check_fn(response):
                failures.append(Failure(
                    type=ftype,
                    issue=issue,
                    severity=severity
                ))
        except Exception:
            # Never crash the pipeline due to a rule check error
            pass

    critical_count = sum(1 for f in failures if f.severity == "critical")
    warning_count  = sum(1 for f in failures if f.severity == "warning")
    total_rules    = len(rules)

    # Score: 1.0 = all pass, 0.0 = all critical fail
    score = max(0.0, 1.0 - (critical_count * 0.3) - (warning_count * 0.1))
    passed = critical_count == 0

    return EvalResult(passed=passed, failures=failures, score=round(score, 2))


def print_eval_result(result: EvalResult) -> None:
    """Print evaluation result to terminal."""
    status = "PASS" if result.passed else "FAIL"
    print(f"\n[EVAL] {status} — score: {result.score:.2f}")

    if result.failures:
        for f in result.failures:
            icon = "[CRITICAL]" if f.severity == "critical" else "[WARN]"
            print(f"  {icon} [{f.type}] {f.issue}")
    print()
