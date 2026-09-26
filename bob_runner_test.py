"""
bob_runner_test.py - Sub-Task 1 test suite for the Bob Shell integration module.

Covers three layers:
  1. Unit tests  - no Bob install or API key required (error-path and import checks).
  2. Auth test   - requires BOB_API_KEY / BOBSHELL_API_KEY in the environment.
  3. End-to-end  - requires BOB_API_KEY + bob CLI installed (calls real Bob).

Usage:
    # Layer 1 only (anyone, no setup needed):
    python bob_runner_test.py --unit

    # Layers 1 + 2 (needs API key in .env or environment):
    python bob_runner_test.py --auth

    # All layers including real Bob calls (needs key + bob installed):
    python bob_runner_test.py --e2e

    # Default (no flag) runs layers 1 + 2:
    python bob_runner_test.py

Setup for teammates:
    1. Copy .env.example to .env and fill in BOB_API_KEY.
    2. Install Bob Shell:  powershell -c "irm https://bob.ibm.com/download/bobshell.ps1 | iex"
    3. Confirm install:    bob --version
    4. Run all tests:      python bob_runner_test.py --e2e
"""

import csv
import io
import os
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env if present (so teammates just need a filled-in .env file)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on env vars being set manually

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
args = set(sys.argv[1:])
RUN_UNIT = True                          # always on
RUN_AUTH = "--unit" not in args          # on unless --unit only
RUN_E2E  = "--e2e" in args              # only when explicitly requested

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
passed = 0
failed = 0


def ok(name: str) -> None:
    global passed
    passed += 1
    print(f"  [PASS] {name}")


def fail(name: str, reason: str) -> None:
    global failed
    failed += 1
    print(f"  [FAIL] {name}: {reason}")


def section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


# ---------------------------------------------------------------------------
# Layer 1 - Unit tests (no key, no bob binary needed)
# ---------------------------------------------------------------------------

section("Layer 1 - Unit tests (no credentials or bob install required)")

# 1.1 All public symbols are importable
try:
    from bob_runner import (
        BobCredentialError,
        BobUnreachableError,
        BobTaskError,
        InvalidRepoError,
        MalformedLogError,
        get_bob_api_key,
        run_bob_onboarding,
        run_bob_triage,
    )
    ok("all public symbols importable from bob_runner")
except ImportError as exc:
    fail("all public symbols importable from bob_runner", str(exc))
    print("\nFATAL: cannot import bob_runner - stopping.\n")
    sys.exit(1)

# 1.2 Exception hierarchy
for cls in (BobCredentialError, BobUnreachableError, BobTaskError,
            InvalidRepoError, MalformedLogError):
    try:
        assert issubclass(cls, Exception)
        ok(f"{cls.__name__} is an Exception subclass")
    except AssertionError:
        fail(f"{cls.__name__} is an Exception subclass", "not a subclass of Exception")

# 1.3 BobTaskError carries stderr_tail
try:
    e = BobTaskError("something went wrong", stderr_tail="last\nlines")
    assert e.stderr_tail == "last\nlines"
    assert str(e) == "something went wrong"
    ok("BobTaskError.stderr_tail attribute present and correct")
except Exception as exc:  # noqa: BLE001
    fail("BobTaskError.stderr_tail attribute present and correct", str(exc))

# 1.4 InvalidRepoError on non-https/git@ URL
try:
    run_bob_onboarding("ftp://bad-url.com/repo")
    fail("InvalidRepoError on bad URL scheme", "no exception raised")
except InvalidRepoError as exc:
    ok(f"InvalidRepoError on bad URL scheme")
except Exception as exc:  # noqa: BLE001
    fail("InvalidRepoError on bad URL scheme", f"wrong exception: {type(exc).__name__}: {exc}")

# 1.5 InvalidRepoError on bare string (no scheme)
try:
    run_bob_onboarding("github.com/foo/bar")
    fail("InvalidRepoError on bare URL (no scheme)", "no exception raised")
except InvalidRepoError:
    ok("InvalidRepoError on bare URL (no scheme)")
except Exception as exc:  # noqa: BLE001
    fail("InvalidRepoError on bare URL (no scheme)", f"wrong exception: {type(exc).__name__}: {exc}")

# 1.6 MalformedLogError on empty bytes
try:
    run_bob_triage(b"", b"")
    fail("MalformedLogError on empty CSV bytes", "no exception raised")
except MalformedLogError:
    ok("MalformedLogError on empty CSV bytes")
except Exception as exc:  # noqa: BLE001
    fail("MalformedLogError on empty CSV bytes", f"wrong exception: {type(exc).__name__}: {exc}")

# 1.7 MalformedLogError on CSV missing iteration column
try:
    run_bob_triage(b"train_loss,val_loss,lr\n0.5,0.6,0.001\n", b"")
    fail("MalformedLogError on CSV missing 'iteration'", "no exception raised")
except MalformedLogError:
    ok("MalformedLogError on CSV missing 'iteration'")
except Exception as exc:  # noqa: BLE001
    fail("MalformedLogError on CSV missing 'iteration'", f"wrong exception: {type(exc).__name__}: {exc}")

# 1.8 MalformedLogError on completely non-CSV bytes
try:
    run_bob_triage(b"\x00\x01\x02", b"")
    fail("MalformedLogError on binary garbage", "no exception raised")
except (MalformedLogError, BobCredentialError):
    # Either is acceptable: MalformedLogError if header parse fails,
    # BobCredentialError if the CSV happens to parse (unlikely with null bytes)
    ok("MalformedLogError or BobCredentialError on binary garbage")
except Exception as exc:  # noqa: BLE001
    fail("MalformedLogError on binary garbage", f"wrong exception: {type(exc).__name__}: {exc}")

# 1.9 Credential check fires before any subprocess (when key is missing)
saved = os.environ.pop("BOB_API_KEY", None)
saved_legacy = os.environ.pop("BOBSHELL_API_KEY", None)
try:
    run_bob_triage(b"iteration,train_loss\n1,0.5\n", b"")
    fail("BobCredentialError when no key in env", "no exception raised")
except BobCredentialError:
    ok("BobCredentialError raised before subprocess when key absent")
except Exception as exc:  # noqa: BLE001
    fail("BobCredentialError when no key in env", f"wrong exception: {type(exc).__name__}: {exc}")
finally:
    if saved is not None:
        os.environ["BOB_API_KEY"] = saved
    if saved_legacy is not None:
        os.environ["BOBSHELL_API_KEY"] = saved_legacy

# ---------------------------------------------------------------------------
# Layer 2 - Auth test (key must be present)
# ---------------------------------------------------------------------------

if RUN_AUTH:
    section("Layer 2 - Auth test (requires BOB_API_KEY in environment or .env)")

    try:
        key = get_bob_api_key()
        if key.startswith("bob_prod_") or len(key) > 20:
            ok(f"get_bob_api_key() returns a plausible key ({key[:20]}...)")
        else:
            fail("get_bob_api_key() returns a plausible key", f"key looks too short: {key!r}")
    except BobCredentialError as exc:
        fail("get_bob_api_key() returns a plausible key",
             "BOB_API_KEY not set - add it to .env or export it in your shell")
    except Exception as exc:  # noqa: BLE001
        fail("get_bob_api_key() returns a plausible key", str(exc))

# ---------------------------------------------------------------------------
# Layer 3 - End-to-end (requires key + bob binary)
# ---------------------------------------------------------------------------

if RUN_E2E:
    section("Layer 3 - End-to-end (requires BOB_API_KEY + bob installed)")

    # 3.1 bob binary is reachable
    import subprocess
    try:
        r = subprocess.run(
            ["bob.cmd" if os.name == "nt" else "bob", "--version"],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode == 0:
            ok(f"bob binary reachable (version: {r.stdout.strip().splitlines()[0]})")
        else:
            fail("bob binary reachable", f"exit {r.returncode}: {r.stderr.strip()[:100]}")
    except FileNotFoundError:
        fail("bob binary reachable",
             "bob not found on PATH - install via: powershell -c \"irm https://bob.ibm.com/download/bobshell.ps1 | iex\"")
    except Exception as exc:  # noqa: BLE001
        fail("bob binary reachable", str(exc))

    # 3.2 run_bob_triage with sample log returns non-empty string + non-empty list
    log_path = Path("logs/train_log.csv")
    cfg_path = Path("logs/run_config.yaml")
    if not log_path.exists():
        fail("run_bob_triage returns (str, list) on sample log",
             f"{log_path} not found - run from the project root")
    else:
        try:
            print("  [INFO] Running Bob triage (may take several minutes)...")
            triage_md, anomalies = run_bob_triage(
                log_path.read_bytes(),
                cfg_path.read_bytes() if cfg_path.exists() else b"",
            )
            assert isinstance(triage_md, str) and len(triage_md) > 100, \
                f"triage_md too short ({len(triage_md)} chars)"
            assert isinstance(anomalies, list) and len(anomalies) > 0, \
                f"anomalies empty or not a list: {type(anomalies)}"
            # Spot-check schema: every item has the 14 required keys
            required_keys = {
                "id", "type", "detector", "severity",
                "epoch_start", "epoch_end", "iteration_start", "iteration_end",
                "evidence", "finding", "reasoning",
                "suggested_fix", "fix_status", "source_report",
            }
            for i, item in enumerate(anomalies):
                missing = required_keys - set(item.keys())
                assert not missing, f"anomaly[{i}] missing keys: {missing}"
            ok(f"run_bob_triage returns valid (str, list) - {len(anomalies)} anomalies, "
               f"all {len(required_keys)} schema keys present")
        except (BobCredentialError, BobUnreachableError) as exc:
            fail("run_bob_triage returns (str, list) on sample log", str(exc))
        except Exception as exc:  # noqa: BLE001
            fail("run_bob_triage returns (str, list) on sample log",
                 f"{type(exc).__name__}: {exc}\n" + traceback.format_exc())

    # 3.3 run_bob_onboarding with a small public repo returns non-empty string
    TEST_REPO = "https://github.com/octocat/Hello-World"
    try:
        print(f"  [INFO] Running Bob onboarding on {TEST_REPO} (may take several minutes)...")
        report = run_bob_onboarding(TEST_REPO)
        assert isinstance(report, str) and len(report) > 100, \
            f"report too short ({len(report)} chars)"
        ok(f"run_bob_onboarding returns non-empty string ({len(report)} chars)")
    except (BobCredentialError, BobUnreachableError) as exc:
        fail("run_bob_onboarding returns non-empty string", str(exc))
    except Exception as exc:  # noqa: BLE001
        fail("run_bob_onboarding returns non-empty string",
             f"{type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = passed + failed
print(f"\n{'=' * 60}")
print(f"  {passed}/{total} passed", end="")
if not RUN_AUTH:
    print("  (layer 1 only - run with --auth or --e2e for more)", end="")
elif not RUN_E2E:
    print("  (layers 1-2 - run with --e2e for full end-to-end)", end="")
print()
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
