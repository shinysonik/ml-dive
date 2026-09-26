"""
Unit tests for bob_runner.py — exercises all five error paths and the happy
path of the public API without needing Bob Shell installed or a real API key.

Run with:
    python tests/test_bob_runner.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure the project root is on sys.path so we can import bob_runner directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import bob_runner
from bob_runner import (
    BobCredentialError,
    BobTaskError,
    BobUnreachableError,
    InvalidRepoError,
    MalformedLogError,
    get_bob_api_key,
    run_bob_onboarding,
    run_bob_triage,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_results: list[tuple[str, str]] = []


def check(name: str, fn):
    try:
        fn()
        _results.append((PASS, name))
    except Exception as exc:  # noqa: BLE001
        _results.append((FAIL, f"{name} — {exc}"))


# ---------------------------------------------------------------------------
# get_bob_api_key
# ---------------------------------------------------------------------------

def test_credential_error_when_no_key():
    with patch.dict(os.environ, {}, clear=True):
        # Also patch st.secrets to be unavailable.
        with patch.dict(sys.modules, {"streamlit": None}):
            try:
                get_bob_api_key()
                raise AssertionError("Should have raised BobCredentialError")
            except BobCredentialError:
                pass  # expected


def test_credential_from_env():
    with patch.dict(os.environ, {"BOBSHELL_API_KEY": "test-key-123"}):
        key = get_bob_api_key()
        assert key == "test-key-123", f"Expected 'test-key-123', got {key!r}"


def test_credential_from_bob_api_key_env():
    with patch.dict(os.environ, {"BOB_API_KEY": "another-key"}, clear=False):
        key = get_bob_api_key()
        assert key == "another-key", f"Expected 'another-key', got {key!r}"


# ---------------------------------------------------------------------------
# run_bob_onboarding — InvalidRepoError on bad URL
# ---------------------------------------------------------------------------

def test_invalid_repo_url_no_scheme():
    try:
        run_bob_onboarding("not-a-url")
        raise AssertionError("Should have raised InvalidRepoError")
    except InvalidRepoError as exc:
        assert "https://" in str(exc) or "git@" in str(exc)


def test_invalid_repo_url_http_only():
    """http:// (not https://) must also raise InvalidRepoError."""
    try:
        run_bob_onboarding("http://github.com/owner/repo")
        raise AssertionError("Should have raised InvalidRepoError")
    except InvalidRepoError:
        pass  # expected


# ---------------------------------------------------------------------------
# run_bob_onboarding — InvalidRepoError when git clone fails
# ---------------------------------------------------------------------------

def test_invalid_repo_when_clone_fails():
    failed_clone = MagicMock()
    failed_clone.returncode = 128
    failed_clone.stderr = "Repository not found."

    with patch.dict(os.environ, {"BOBSHELL_API_KEY": "dummy"}):
        with patch("bob_runner.subprocess.run", return_value=failed_clone):
            try:
                run_bob_onboarding("https://github.com/nonexistent/repo")
                raise AssertionError("Should have raised InvalidRepoError")
            except InvalidRepoError as exc:
                assert "clone" in str(exc).lower() or "repository" in str(exc).lower()


# ---------------------------------------------------------------------------
# run_bob_onboarding — BobUnreachableError when bob not on PATH
# ---------------------------------------------------------------------------

def test_bob_unreachable_onboarding():
    ok_clone = MagicMock()
    ok_clone.returncode = 0
    ok_clone.stderr = ""

    def side_effect(cmd, **kwargs):
        # First call is git clone — succeeds; second call is bob — raises FileNotFoundError.
        if isinstance(cmd, list) and cmd[0] == "git":
            return ok_clone
        raise FileNotFoundError("bob not found")

    with patch.dict(os.environ, {"BOBSHELL_API_KEY": "dummy"}):
        with patch("bob_runner.subprocess.run", side_effect=side_effect):
            try:
                run_bob_onboarding("https://github.com/owner/repo")
                raise AssertionError("Should have raised BobUnreachableError")
            except BobUnreachableError:
                pass  # expected


# ---------------------------------------------------------------------------
# run_bob_onboarding — BobTaskError on non-zero bob exit
# ---------------------------------------------------------------------------

def test_bob_task_error_nonzero_exit_onboarding():
    ok_clone = MagicMock()
    ok_clone.returncode = 0
    ok_clone.stderr = ""

    bad_bob = MagicMock()
    bad_bob.returncode = 1
    bad_bob.stderr = "line1\nline2\nerror on last line"
    bad_bob.stdout = ""

    def side_effect(cmd, **kwargs):
        if isinstance(cmd, list) and cmd[0] == "git":
            return ok_clone
        return bad_bob

    with patch.dict(os.environ, {"BOBSHELL_API_KEY": "dummy"}):
        with patch("bob_runner.subprocess.run", side_effect=side_effect):
            try:
                run_bob_onboarding("https://github.com/owner/repo")
                raise AssertionError("Should have raised BobTaskError")
            except BobTaskError as exc:
                assert exc.stderr_tail  # stderr tail must be attached
                assert "error on last line" in exc.stderr_tail


# ---------------------------------------------------------------------------
# run_bob_onboarding — BobTaskError when ONBOARDING_REPORT.md absent
# ---------------------------------------------------------------------------

def test_bob_task_error_missing_report_onboarding():
    """Bob exits 0 but does not write ONBOARDING_REPORT.md → BobTaskError."""
    ok_clone = MagicMock()
    ok_clone.returncode = 0
    ok_clone.stderr = ""

    ok_bob = MagicMock()
    ok_bob.returncode = 0
    ok_bob.stderr = ""
    ok_bob.stdout = ""

    def side_effect(cmd, **kwargs):
        if isinstance(cmd, list) and cmd[0] == "git":
            return ok_clone
        return ok_bob

    with patch.dict(os.environ, {"BOBSHELL_API_KEY": "dummy"}):
        with patch("bob_runner.subprocess.run", side_effect=side_effect):
            try:
                run_bob_onboarding("https://github.com/owner/repo")
                raise AssertionError("Should have raised BobTaskError")
            except BobTaskError as exc:
                assert "ONBOARDING_REPORT.md" in str(exc)


# ---------------------------------------------------------------------------
# run_bob_onboarding — happy path (Bob writes the report file)
# ---------------------------------------------------------------------------

def test_onboarding_happy_path():
    """Bob exits 0 and writes ONBOARDING_REPORT.md → function returns its content."""
    ok_clone = MagicMock()
    ok_clone.returncode = 0
    ok_clone.stderr = ""

    ok_bob = MagicMock()
    ok_bob.returncode = 0
    ok_bob.stderr = ""
    ok_bob.stdout = ""

    report_content = "# Onboarding Report\n\nAll good.\n"

    def side_effect(cmd, **kwargs):
        if isinstance(cmd, list) and cmd[0] == "git":
            return ok_clone
        # Bob call (shell=True, so cmd is a string) — write the report file.
        # cwd is always passed as str(workspace) by _run_bob_in_dir.
        cwd = kwargs.get("cwd")
        if cwd:
            (Path(cwd) / "ONBOARDING_REPORT.md").write_text(report_content)
        return ok_bob

    with patch.dict(os.environ, {"BOBSHELL_API_KEY": "dummy"}):
        with patch("bob_runner.subprocess.run", side_effect=side_effect):
            result = run_bob_onboarding("https://github.com/owner/repo")
            assert result == report_content, f"Unexpected content: {result!r}"


# ---------------------------------------------------------------------------
# run_bob_triage — MalformedLogError on empty CSV
# ---------------------------------------------------------------------------

def test_malformed_log_empty():
    try:
        run_bob_triage(b"", b"")
        raise AssertionError("Should have raised MalformedLogError")
    except MalformedLogError as exc:
        assert "empty" in str(exc).lower() or "iteration" in str(exc).lower()


def test_malformed_log_missing_iteration_column():
    csv = b"epoch,loss,val_loss\n1,1.0,1.1\n"
    try:
        run_bob_triage(csv, b"")
        raise AssertionError("Should have raised MalformedLogError")
    except MalformedLogError as exc:
        assert "iteration" in str(exc).lower()


# ---------------------------------------------------------------------------
# run_bob_triage — BobTaskError when TRIAGE_REPORT.md absent
# ---------------------------------------------------------------------------

def test_bob_task_error_missing_triage_report():
    ok_bob = MagicMock()
    ok_bob.returncode = 0
    ok_bob.stderr = ""
    ok_bob.stdout = ""

    csv = b"iteration,train_loss,val_loss,lr\n1,1.0,1.1,0.001\n"

    with patch.dict(os.environ, {"BOBSHELL_API_KEY": "dummy"}):
        with patch("bob_runner.subprocess.run", return_value=ok_bob):
            try:
                run_bob_triage(csv, b"")
                raise AssertionError("Should have raised BobTaskError")
            except BobTaskError as exc:
                assert "TRIAGE_REPORT.md" in str(exc)


# ---------------------------------------------------------------------------
# run_bob_triage — BobTaskError when anomalies.json absent
# ---------------------------------------------------------------------------

def test_bob_task_error_missing_anomalies():
    ok_bob = MagicMock()
    ok_bob.returncode = 0
    ok_bob.stderr = ""
    ok_bob.stdout = ""

    triage_content = "# Triage Report\n\nDone.\n"
    csv = b"iteration,train_loss,val_loss,lr\n1,1.0,1.1,0.001\n"

    def side_effect(cmd, **kwargs):
        cwd = kwargs.get("cwd")
        if cwd:
            (Path(cwd) / "TRIAGE_REPORT.md").write_text(triage_content)
        # anomalies.json intentionally not written
        return ok_bob

    with patch.dict(os.environ, {"BOBSHELL_API_KEY": "dummy"}):
        with patch("bob_runner.subprocess.run", side_effect=side_effect):
            try:
                run_bob_triage(csv, b"")
                raise AssertionError("Should have raised BobTaskError")
            except BobTaskError as exc:
                assert "anomalies.json" in str(exc)


# ---------------------------------------------------------------------------
# run_bob_triage — happy path
# ---------------------------------------------------------------------------

def test_triage_happy_path():
    ok_bob = MagicMock()
    ok_bob.returncode = 0
    ok_bob.stderr = ""
    ok_bob.stdout = ""

    triage_content = "# Triage Report\n\nDone.\n"
    anomalies_data = [{"type": "nan_loss", "iteration_start": 100, "iteration_end": 200}]
    csv = b"iteration,train_loss,val_loss,lr\n1,1.0,1.1,0.001\n"

    def side_effect(cmd, **kwargs):
        cwd = kwargs.get("cwd")
        if cwd:
            (Path(cwd) / "TRIAGE_REPORT.md").write_text(triage_content)
            analysis_dir = Path(cwd) / "analysis"
            analysis_dir.mkdir(exist_ok=True)
            (analysis_dir / "anomalies.json").write_text(json.dumps(anomalies_data))
        return ok_bob

    with patch.dict(os.environ, {"BOBSHELL_API_KEY": "dummy"}):
        with patch("bob_runner.subprocess.run", side_effect=side_effect):
            triage_md, anomalies = run_bob_triage(csv, b"")
            assert triage_md == triage_content
            assert anomalies == anomalies_data


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    ("get_bob_api_key: raises BobCredentialError when no key set",       test_credential_error_when_no_key),
    ("get_bob_api_key: reads BOBSHELL_API_KEY from env",                  test_credential_from_env),
    ("get_bob_api_key: reads BOB_API_KEY from env",                       test_credential_from_bob_api_key_env),
    ("run_bob_onboarding: InvalidRepoError on bare URL",                  test_invalid_repo_url_no_scheme),
    ("run_bob_onboarding: InvalidRepoError on http:// URL",               test_invalid_repo_url_http_only),
    ("run_bob_onboarding: InvalidRepoError when git clone fails",         test_invalid_repo_when_clone_fails),
    ("run_bob_onboarding: BobUnreachableError when bob not on PATH",      test_bob_unreachable_onboarding),
    ("run_bob_onboarding: BobTaskError on non-zero bob exit (stderr_tail attached)", test_bob_task_error_nonzero_exit_onboarding),
    ("run_bob_onboarding: BobTaskError when ONBOARDING_REPORT.md absent", test_bob_task_error_missing_report_onboarding),
    ("run_bob_onboarding: happy path returns report content",             test_onboarding_happy_path),
    ("run_bob_triage: MalformedLogError on empty CSV",                    test_malformed_log_empty),
    ("run_bob_triage: MalformedLogError when 'iteration' column missing", test_malformed_log_missing_iteration_column),
    ("run_bob_triage: BobTaskError when TRIAGE_REPORT.md absent",        test_bob_task_error_missing_triage_report),
    ("run_bob_triage: BobTaskError when anomalies.json absent",           test_bob_task_error_missing_anomalies),
    ("run_bob_triage: happy path returns (triage_md, anomalies_dict)",    test_triage_happy_path),
]


if __name__ == "__main__":
    print("=" * 70)
    print("bob_runner unit tests")
    print("=" * 70)

    for name, fn in ALL_TESTS:
        check(name, fn)

    print()
    passed = sum(1 for r, _ in _results if "PASS" in r)
    total = len(_results)
    for status, name in _results:
        print(f"  [{status}] {name}")
    print()
    print(f"{'=' * 70}")
    print(f"  {passed}/{total} passed")
    print(f"{'=' * 70}")

    if passed < total:
        sys.exit(1)
