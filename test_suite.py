#!/usr/bin/env python3
# test_suite.py
# v5 — adds timeout=180 to subprocess.run() in run_test(); hung Ollama calls
#      now raise subprocess.TimeoutExpired, which is caught and returned as a
#      failed EvalResult with a single critical Failure instead of hanging
#
# Stable result: 8/9 on the reference setup. Test 9 is probabilistic — retry
# fires when the model reaches for eval() on first attempt, does not fire when
# it proactively uses json.loads(). Both outcomes are correct pipeline behavior.
# The constraint is working. 8/9 is accepted as the stable baseline.
#
# Changes from v4:
#   - run_test() subprocess.run() call now passes timeout=180. Wrapped in
#     try/except subprocess.TimeoutExpired — on timeout, prints which prompt
#     timed out and returns a failed EvalResult with a single Failure of
#     type="integration", severity="critical", plus retry_fired=False.
#
# Changes from v3:
#   - test 7 ("write a python function to parse a json file") moved from TESTS
#     to RETRY_TESTS — confirmed retry case: model reaches for eval() on JSON
#     parsing at temperature=0.7, retry prepends the specific failure reason,
#     model corrects to json.loads() on second attempt
#   - domain_specialist prompts updated for generic project structure
#
# Usage: python3 test_suite.py

import subprocess
from evaluator import evaluate, print_eval_result, EvalResult, Failure

TESTS = [
    ("implement the main handler for the domain_specialist task",                            "domain_specialist"),
    ("implement the core processing class with anomaly detection",                           "domain_specialist"),
    ("write a script to read files from the vault",                                          "coding"),
    ("how do i check if the open-webui docker container is running",                         "linux_admin"),
    ("write a python snippet to call the ollama api and get a response",                     "ai_dev"),
    ("debug the connection issue",                                                           "default"),
    ("implement the websocket handler in the api server",                                    "domain_specialist"),
    ("implement the api — create a new fastapi app instance for the domain_specialist task", "domain_specialist"),
]

# Tests where retry is expected to fire and the final output is expected to pass.
# A test here fails if: retry did not fire, OR final output fails evaluation.
RETRY_TESTS = [
    (
        "write a python function to parse a json file",
        "coding",
        "eval() used for JSON parsing — retry should redirect to json.loads()",
    ),
]


def parse_stdout(stdout: str) -> tuple[str, bool]:
    """
    Parse run_task.py stdout into (final_output, retry_fired).

    Strips the status line ([HH:MM:SS] → model [skill:x]).
    Detects [RETRY] marker and returns only the output after it if present.
    """
    # Strip the status line — always the first non-empty line
    lines = stdout.splitlines()
    content_lines = []
    skip_next_blank = False
    for i, line in enumerate(lines):
        # Status line pattern: starts with [HH:MM:SS]
        if line.startswith("[") and "→" in line and len(line) < 80:
            skip_next_blank = True
            continue
        if skip_next_blank and line.strip() == "":
            skip_next_blank = False
            continue
        content_lines.append(line)

    content = "\n".join(content_lines)

    if "[RETRY]" in content:
        # Split on [RETRY] — take everything after it as the final output
        final_output = content.split("[RETRY]", 1)[1].strip()
        return final_output, True

    return content.strip(), False


def run_test(prompt: str, task: str) -> tuple:
    """
    Run a prompt through the pipeline.
    Returns (eval_result, retry_fired).
    """
    try:
        result = subprocess.run(
            ["python3", "run_task.py", prompt],
            capture_output=True, text=True, timeout=180
        )
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Prompt timed out after 180s: {prompt[:80]}")
        return EvalResult(
            passed=False,
            failures=[Failure(
                type="integration",
                issue=f"subprocess timed out after 180s — prompt: {prompt[:80]}",
                severity="critical",
            )],
            score=0.0,
        ), False
    final_output, retry_fired = parse_stdout(result.stdout + result.stderr)
    return evaluate(final_output, task), retry_fired


if __name__ == "__main__":
    total_tests = len(TESTS) + len(RETRY_TESTS)
    print(f"\nRunning {total_tests} tests...\n" + "─" * 52)
    passed = 0

    # ── Standard tests ────────────────────────────────────────────────────────
    print("\n  Standard tests (retry not expected)\n")
    for i, (prompt, task) in enumerate(TESTS, 1):
        print(f"Test {i} [{task}]: {prompt[:60]}")
        result, retry_fired = run_test(prompt, task)
        print_eval_result(result)
        if retry_fired:
            print(f"  [WARN] Retry fired unexpectedly on test {i}")
        if result.passed:
            passed += 1

    # ── Retry tests ───────────────────────────────────────────────────────────
    print("\n" + "─" * 52)
    print("\n  Retry tests (retry must fire AND final output must pass)\n")
    for i, (prompt, task, description) in enumerate(RETRY_TESTS, 1):
        label = len(TESTS) + i
        print(f"Test {label} [retry/{task}]: {prompt[:60]}")
        print(f"  Expected: {description}")
        result, retry_fired = run_test(prompt, task)
        print_eval_result(result)

        if not retry_fired:
            print(f"  RETRY DID NOT FIRE — first attempt passed (constraint may be too loose)")
        else:
            print(f"  Retry fired")

        # Pass only if retry fired AND final output passes evaluation
        if retry_fired and result.passed:
            passed += 1

    print("─" * 52)
    print(f"Results: {passed}/{total_tests} passed\n")
