"""
bob_runner.py — IBM Bob Shell integration module for ML-Dive.

Public API:
    run_bob_onboarding(repo_url: str) -> str
    run_bob_triage(log_csv_bytes: bytes, run_config_bytes: bytes) -> tuple[str, dict]
    get_bob_api_key() -> str

All Bob Shell interaction is isolated here.  ml_dive.py imports only the two
run_* functions and the named exception classes.
"""

import csv
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Named exception classes
# ---------------------------------------------------------------------------

class BobCredentialError(Exception):
    """Raised when BOB_API_KEY is not found in secrets or environment."""


class BobUnreachableError(Exception):
    """Raised when the `bob` executable is not found on PATH."""


class BobTaskError(Exception):
    """Raised when Bob exits with a non-zero code or does not produce expected output files.

    Attributes:
        stderr_tail (str): Last 20 lines of Bob's stderr, for display in a UI expander.
    """

    def __init__(self, message: str, stderr_tail: str = ""):
        super().__init__(message)
        self.stderr_tail = stderr_tail


class InvalidRepoError(Exception):
    """Raised when a repository URL is invalid or cannot be cloned."""


class MalformedLogError(Exception):
    """Raised when the training log CSV is empty or missing the `iteration` column."""


# ---------------------------------------------------------------------------
# Credential helper
# ---------------------------------------------------------------------------

def get_bob_api_key() -> str:
    """Return the Bob Shell API key.

    Checks sources in this order:
    1. ``st.secrets["BOB_API_KEY"]`` (Streamlit Cloud context)
    2. ``os.environ["BOB_API_KEY"]`` (local / server context)
    3. ``os.environ["BOBSHELL_API_KEY"]`` (legacy fallback)

    Raises:
        BobCredentialError: if the key is not found in any source.
    """
    # Attempt Streamlit secrets (only available when running inside Streamlit).
    try:
        import streamlit as st  # noqa: PLC0415 — deferred import intentional
        key = st.secrets.get("BOB_API_KEY") or st.secrets.get("BOBSHELL_API_KEY")
        if key:
            return key
    except Exception:  # noqa: BLE001 — covers ImportError and Streamlit bootstrap errors
        pass

    # Fall back to environment variables.
    key = os.environ.get("BOB_API_KEY") or os.environ.get("BOBSHELL_API_KEY")
    if key:
        return key

    raise BobCredentialError(
        "BOB_API_KEY not found. Set it as an environment variable (local) "
        "or add it to Streamlit Cloud secrets."
    )


# ---------------------------------------------------------------------------
# Shared subprocess helper
# ---------------------------------------------------------------------------

# On Windows, bob is installed as bob.cmd by npm; on POSIX it is just `bob`.
_BOB_CMD = "bob.cmd" if os.name == "nt" else "bob"

_PROMPT_FILENAME = "bob_prompt.txt"


def _run_bob_in_dir(workspace: Path, prompt: str, timeout_seconds: int) -> None:
    """Run Bob Shell inside *workspace* using *prompt* as the task.

    The prompt is written to a file inside the workspace and fed to Bob via
    stdin redirect (``bob run ... < prompt_file``).  This avoids Windows
    command-line length limits that occur when passing large prompts as
    positional arguments.

    Args:
        workspace: Absolute path to the temp workspace directory.
        prompt: Full prompt text to pass to Bob.
        timeout_seconds: Hard wall-clock limit for the Bob subprocess.

    Raises:
        BobCredentialError: if the API key is missing (checked before launch).
        BobUnreachableError: if the `bob` binary is not on PATH.
        BobTaskError: if Bob exits with a non-zero exit code.
    """
    api_key = get_bob_api_key()

    prompt_file = workspace / _PROMPT_FILENAME
    prompt_file.write_text(prompt, encoding="utf-8")

    env = os.environ.copy()
    env["BOB_API_KEY"] = api_key

    # Use shell=True so the < stdin redirect is handled by the shell.
    # Quoting workspace and prompt_file handles spaces in temp paths.
    cmd = (
        f'{_BOB_CMD} run --trust --accept-license -w "{workspace}"'
        f' < "{prompt_file}"'
    )

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(workspace),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise BobUnreachableError(
            "Bob Shell (`bob`) is not installed or not on PATH. "
            "Install it from the Bob portal and ensure the `bob` command is available."
        ) from exc

    if result.returncode != 0:
        stderr_lines = (result.stderr or "").splitlines()
        stderr_tail = "\n".join(stderr_lines[-20:])
        raise BobTaskError(
            f"Bob exited with code {result.returncode}. "
            "The task may have timed out or Bob's response was incomplete.",
            stderr_tail=stderr_tail,
        )


# ---------------------------------------------------------------------------
# Public function: run_bob_onboarding
# ---------------------------------------------------------------------------

_ONBOARDING_PROMPT_PATH = Path(__file__).parent / "prompts" / "onboarding_auto.md"
_ONBOARDING_TIMEOUT = 600  # 10 minutes


def run_bob_onboarding(repo_url: str) -> str:
    """Clone *repo_url* and run Bob's onboarding analysis on it.

    Args:
        repo_url: Public Git repository URL (``https://`` or ``git@`` prefix).

    Returns:
        Content of ``ONBOARDING_REPORT.md`` as a string.

    Raises:
        InvalidRepoError: if the URL is malformed or the clone fails.
        BobCredentialError: if the API key is missing.
        BobUnreachableError: if `bob` is not on PATH.
        BobTaskError: if Bob does not produce ``ONBOARDING_REPORT.md``.
    """
    # 1. Validate URL format.
    if not (repo_url.startswith("https://") or repo_url.startswith("git@")):
        raise InvalidRepoError(
            f"Invalid repository URL: {repo_url!r}. "
            "The URL must start with 'https://' or 'git@'."
        )

    prompt = _ONBOARDING_PROMPT_PATH.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)

        # 2. Clone the repository (shallow clone for speed).
        clone_result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(workspace)],
            capture_output=True,
            text=True,
        )
        if clone_result.returncode != 0:
            detail = clone_result.stderr.strip()
            raise InvalidRepoError(
                f"Could not clone the repository: {repo_url!r}. "
                "Check that the URL is correct and the repository is public."
                + (f"\n\nGit output:\n{detail}" if detail else "")
            )

        # 3. Run Bob.
        _run_bob_in_dir(workspace, prompt, timeout_seconds=_ONBOARDING_TIMEOUT)

        # 4. Read the output file.
        report_path = workspace / "ONBOARDING_REPORT.md"
        if not report_path.exists():
            raise BobTaskError(
                "Bob ran but did not produce ONBOARDING_REPORT.md.",
                stderr_tail="",
            )

        return report_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Public function: run_bob_triage
# ---------------------------------------------------------------------------

_DEBUGGING_PROMPT_PATH = Path(__file__).parent / "prompts" / "debugging_auto.md"
_TRIAGE_TIMEOUT = 900  # 15 minutes


def run_bob_triage(log_csv_bytes: bytes, run_config_bytes: bytes) -> tuple:
    """Run Bob's triage analysis on a training log CSV and optional run config.

    Args:
        log_csv_bytes: Raw bytes of ``train_log.csv``.
        run_config_bytes: Raw bytes of ``run_config.yaml`` (may be ``b""``).

    Returns:
        ``(triage_md_str, anomalies_dict)`` — the triage report as a string
        and the parsed ``anomalies.json`` as a Python dict/list.

    Raises:
        MalformedLogError: if the CSV is empty or missing the ``iteration`` column.
        BobCredentialError: if the API key is missing.
        BobUnreachableError: if `bob` is not on PATH.
        BobTaskError: if Bob does not produce the expected output files.
    """
    # 1. Validate the CSV header.
    try:
        text = log_csv_bytes.decode("utf-8", errors="replace")
        reader = csv.reader(io.StringIO(text))
        header = next(reader)
    except StopIteration:
        raise MalformedLogError(
            "The training log CSV is empty. "
            "It must contain at least a header row with an 'iteration' column."
        )
    except Exception as exc:  # noqa: BLE001
        raise MalformedLogError(
            f"Could not parse the training log CSV: {exc}"
        ) from exc

    if "iteration" not in [col.strip().lower() for col in header]:
        raise MalformedLogError(
            "The training log CSV is missing the 'iteration' column. "
            "Expected columns include at least: iteration, train_loss, val_loss, lr."
        )

    prompt = _DEBUGGING_PROMPT_PATH.read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)

        # 2. Write inputs into the workspace.
        logs_dir = workspace / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "train_log.csv").write_bytes(log_csv_bytes)
        if run_config_bytes:
            (logs_dir / "run_config.yaml").write_bytes(run_config_bytes)

        # 3. Run Bob.
        _run_bob_in_dir(workspace, prompt, timeout_seconds=_TRIAGE_TIMEOUT)

        # 4. Read TRIAGE_REPORT.md.
        triage_path = workspace / "TRIAGE_REPORT.md"
        if not triage_path.exists():
            raise BobTaskError(
                "Bob ran but did not produce TRIAGE_REPORT.md.",
                stderr_tail="",
            )
        triage_md = triage_path.read_text(encoding="utf-8")

        # 5. Read analysis/anomalies.json.
        anomalies_path = workspace / "analysis" / "anomalies.json"
        if not anomalies_path.exists():
            raise BobTaskError(
                "Bob ran but did not produce analysis/anomalies.json.",
                stderr_tail="",
            )
        try:
            anomalies = json.loads(anomalies_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BobTaskError(
                f"Bob produced analysis/anomalies.json but it is not valid JSON: {exc}",
                stderr_tail="",
            ) from exc

        return triage_md, anomalies
