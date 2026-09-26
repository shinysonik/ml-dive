"""Edge case handling for ML-Dive.

Every user-facing error message lives here. Nothing else in the app should
ever show a raw exception or a Python traceback.

Three layers:
1. Exceptions  - named error classes raised by validators and by Bob calls.
2. Validators  - pure functions, no Streamlit imports, testable in isolation.
3. UI          - Streamlit helpers that render the right message for an error.

The user-facing text follows docs/ERROR_MESSAGES.md. If you add a new error,
add its message there first, then add the class and the UI mapping here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ============================================================
# 1. Exceptions
# ============================================================

class UserFacingError(Exception):
    """Base class. Anything raised as a UserFacingError is safe to show.

    The `headline`, `explanation`, and `help_key` attributes drive the UI.
    `detail` is optional technical info shown inside a collapsed expander.
    """

    headline = "Something went wrong."
    explanation = ""
    help_key = ""

    def __init__(self, detail: str = ""):
        super().__init__(detail or self.headline)
        self.detail = detail


class InvalidRepoUrlError(UserFacingError):
    headline = "That does not look like a GitHub URL."
    explanation = "A valid URL looks like https://github.com/owner/repo."
    help_key = "invalid_repo_url"


class RepoNotFoundError(UserFacingError):
    headline = "This repository could not be found."
    explanation = "It may have been deleted or renamed."
    help_key = "repo_not_found"


class RepoPrivateError(UserFacingError):
    headline = "This repository is private."
    explanation = "Bob cannot read it without access."
    help_key = "repo_private"


class RepoEmptyError(UserFacingError):
    headline = "This repository appears to be empty."
    explanation = "Bob found no code files to analyze."
    help_key = "repo_empty"


class RepoTooLargeError(UserFacingError):
    headline = "This repository is too large for Bob to analyze in one pass."
    explanation = "Try pointing to a subdirectory, or clone locally and upload a folder."
    help_key = "repo_too_large"


class RepoNoCodeError(UserFacingError):
    headline = "Bob did not find any code in this repository."
    explanation = "It contains only documentation or data files."
    help_key = "repo_no_code"


class LogMissingError(UserFacingError):
    headline = "No training log was uploaded."
    explanation = "Triage needs a log file with iteration, epoch, train_loss, val_loss, lr."
    help_key = "log_missing"


class LogEmptyError(UserFacingError):
    headline = "The training log is empty."
    explanation = "It has a header but no data rows."
    help_key = "log_empty"


class LogMalformedError(UserFacingError):
    headline = "The training log could not be read."
    explanation = "It is missing columns that are required for triage."
    help_key = "log_malformed"


class LogTooLargeError(UserFacingError):
    headline = "This log is too large to process."
    explanation = "Downsample or split the log into smaller runs."
    help_key = "log_too_large"


class LogNoFiniteError(UserFacingError):
    headline = "This training run never produced a finite loss."
    explanation = "Every value in train_loss and val_loss is NaN or Inf from iteration 1."
    help_key = "log_no_finite"


class LogConfigMismatchError(UserFacingError):
    headline = "The log and the run config do not match."
    explanation = "Make sure both files come from the same training run."
    help_key = "log_config_mismatch"


class BobCredentialError(UserFacingError):
    headline = "Bob credentials are missing or invalid."
    explanation = "The app cannot reach Bob without valid credentials."
    help_key = "bob_credentials"


class BobUnreachableError(UserFacingError):
    headline = "Bob is not responding right now."
    explanation = "This is usually temporary."
    help_key = "bob_unreachable"


class BobTaskError(UserFacingError):
    headline = "Bob could not complete the analysis."
    explanation = "Bob ran but did not produce the expected output."
    help_key = "bob_task"


class BobTimeoutError(UserFacingError):
    headline = "Bob is taking longer than expected."
    explanation = "The repository may be very large, or the service may be busy."
    help_key = "bob_timeout"


class BobcoinsExhaustedError(UserFacingError):
    headline = "Your Bob account has no Bobcoins left."
    explanation = "Each analysis costs Bobcoins, and the account has none remaining."
    help_key = "bobcoins"


class RateLimitError(UserFacingError):
    headline = "Too many requests in a short time."
    explanation = "Bob is rate-limiting this account. Wait a minute and retry."
    help_key = "rate_limit"


# ============================================================
# 2. Validators (pure, no Streamlit)
# ============================================================

GITHUB_RE = re.compile(r"^https?://(?:www\.)?github\.com/([^/]+)/([^/#?]+?)(?:\.git)?/?$")

SUPPORTED_LOG_EXTENSIONS = {
    ".csv": "csv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".json": "json",
    ".parquet": "parquet",
}

REQUIRED_LOG_COLUMNS = ["iteration", "train_loss", "val_loss", "lr"]

COLUMN_ALIASES = {
    "iteration": ["step", "global_step", "iter"],
    "epoch": ["ep"],
    "train_loss": ["loss", "training_loss", "train/loss"],
    "val_loss": ["validation_loss", "val/loss", "eval_loss"],
    "lr": ["learning_rate", "learning rate"],
}

MAX_LOG_BYTES = 200 * 1024 * 1024  # 200 MB
MAX_REPO_BYTES = 500 * 1024 * 1024  # 500 MB


def normalize_github_url(raw: str) -> str:
    """Add https:// if missing. Does not validate shape."""
    raw = raw.strip()
    if not raw:
        return raw
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw
    return raw


def parse_github_url(raw: str) -> tuple[str, str]:
    """Return (owner, repo). Raise InvalidRepoUrlError if it does not parse."""
    normalized = normalize_github_url(raw)
    match = GITHUB_RE.match(normalized)
    if not match:
        raise InvalidRepoUrlError(detail=f"Input was: {raw!r}")
    owner, repo = match.group(1), match.group(2)
    return owner, repo


def check_repo_public(owner: str, repo: str, timeout: int = 10) -> None:
    """Check via the GitHub public API whether the repository is public.

    Raises RepoNotFoundError or RepoPrivateError.
    Does not call Bob. Costs no Bobcoins.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        # Network problem - let the caller fall through to Bob
        raise BobUnreachableError(detail=f"GitHub API unreachable: {e}")

    if resp.status_code == 200:
        data = resp.json()
        if data.get("size", 0) * 1024 > MAX_REPO_BYTES:
            raise RepoTooLargeError(detail=f"Repo size: {data.get('size')} KB")
        return
    if resp.status_code == 404:
        raise RepoNotFoundError(detail=f"GET {url} returned 404")
    if resp.status_code == 403:
        # Could be rate limiting on the anonymous API, or a private repo
        # that returns 404. Treat as private so the user sees the helpful message.
        raise RepoPrivateError(detail=f"GET {url} returned 403")


def detect_file_format(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_LOG_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_LOG_EXTENSIONS))
        raise LogMalformedError(
            detail=f"Unsupported extension {ext!r}. Supported: {supported}"
        )
    return SUPPORTED_LOG_EXTENSIONS[ext]


def read_log(raw: bytes, filename: str) -> pd.DataFrame:
    """Read a log file of any supported format into a DataFrame."""
    if not raw:
        raise LogEmptyError()
    if len(raw) > MAX_LOG_BYTES:
        raise LogTooLargeError(detail=f"Size: {len(raw)} bytes")

    fmt = detect_file_format(filename)
    from io import BytesIO

    try:
        if fmt == "csv":
            df = pd.read_csv(BytesIO(raw))
        elif fmt == "jsonl":
            df = pd.read_json(BytesIO(raw), lines=True)
        elif fmt == "json":
            df = pd.read_json(BytesIO(raw))
        elif fmt == "parquet":
            df = pd.read_parquet(BytesIO(raw))
        else:
            raise LogMalformedError(detail=f"Unknown format {fmt!r}")
    except LogMalformedError:
        raise
    except Exception as e:
        raise LogMalformedError(detail=f"pandas could not parse the file: {e}")

    if df.empty:
        raise LogEmptyError()
    return df


def map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Map alternative column names to the canonical ones.

    Returns (renamed_df, list_of_still_missing_columns).
    """
    renamed = df.copy()
    lower_map = {c.lower(): c for c in renamed.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in renamed.columns:
            continue
        for alias in aliases:
            if alias.lower() in lower_map:
                renamed = renamed.rename(columns={lower_map[alias.lower()]: canonical})
                break

    missing = [c for c in REQUIRED_LOG_COLUMNS if c not in renamed.columns]
    return renamed, missing


def validate_log(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a parsed log. Returns the same df if valid, raises otherwise."""
    renamed, missing = map_columns(df)
    if missing:
        raise LogMalformedError(detail=f"Missing columns: {', '.join(missing)}")

    for col in ("train_loss", "val_loss"):
        series = pd.to_numeric(renamed[col], errors="coerce")
        renamed[col] = series
        if not np.isfinite(series).any():
            raise LogNoFiniteError(detail=f"Column {col!r} has no finite values")

    return renamed


def validate_log_config_pair(df: pd.DataFrame, run_config: dict) -> None:
    """Optional check: log iteration range vs config MAX_ITER."""
    if not run_config:
        return
    max_iter = run_config.get("SOLVER", {}).get("MAX_ITER")
    if max_iter is None:
        return
    log_max = int(df["iteration"].max())
    if log_max > max_iter * 1.05:
        raise LogConfigMismatchError(
            detail=f"Log goes to iteration {log_max}, config defines MAX_ITER={max_iter}"
        )


# ============================================================
# 3. UI helpers (Streamlit)
# ============================================================

HELP_BLOCKS = {
    "invalid_repo_url": """
A valid GitHub URL has this shape:

    https://github.com/owner/repo

If you copied the URL from a browser tab, make sure you did not include
extra path like `/tree/main/src`. You can include it, but the owner and
repository names must be the first two parts after github.com.
""",
    "repo_not_found": """
Open the URL in your browser to check it. If the page shows a 404, the
repository has been deleted or renamed. Copy the URL again from GitHub
and paste it back here.
""",
    "repo_private": """
Bob cannot read private repositories without access. Two options:

**Option A - make the repository public temporarily.**

1. Open the repository on GitHub.
2. Click Settings.
3. Scroll to the Danger Zone at the bottom.
4. Click Change visibility, then Change to public.

[Open GitHub docs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility)

**Option B - provide a personal access token.**

1. Go to GitHub, Settings, Developer settings, Personal access tokens.
2. Generate a token with `repo` scope.
3. Paste it in the field below.

[Open GitHub docs](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)
""",
    "repo_empty": """
The onboarding workflow works best with a repository that contains source
code. Empty repositories and repositories with only documentation cannot
be analyzed.
""",
    "repo_too_large": """
Three options:

1. Point to a subdirectory: change the URL to
   `https://github.com/owner/repo/tree/main/src` to analyze only `src`.
2. Clone the repository locally, then upload only the folder you care about.
3. Use a smaller fork if one exists.
""",
    "repo_no_code": """
Bob looks for source files: `.py`, `.js`, `.ts`, `.java`, `.go`, `.rs`,
`.cpp`, `.c`, `.rb`, `.php` and similar. Repositories with only markdown
or data files cannot be analyzed with the onboarding workflow.
""",
    "log_missing": """
When a model trains, it writes a record of each step. That record is the
training log. It usually contains columns like `iteration`, `epoch`,
`train_loss`, `val_loss`, `lr`.

Export it from your training framework and upload the file here.
""",
    "log_empty": """
The file has a header but no data rows. Export the log again from your
training run and check that the output is not empty.

- PyTorch Lightning: check the `csv_logs` folder in your output directory.
- HuggingFace Trainer: look at `trainer_state.json`.
- TensorBoard: export scalars as CSV from the TensorBoard UI.
""",
    "log_malformed": """
Different training frameworks use different column names. The app will
try to map the common ones automatically:

| Your column | Mapped to |
|---|---|
| step, global_step, iter | iteration |
| loss, training_loss | train_loss |
| validation_loss, eval_loss | val_loss |
| learning_rate | lr |

If the mapping fails, rename the columns in your file and upload again.
""",
    "log_too_large": """
Two options:

1. Downsample: keep every tenth row. Most anomalies remain visible at
   lower resolution.
2. Split: upload the first half and the second half as separate runs.
""",
    "log_no_finite": """
This usually means the model was broken from the very start. Common causes:

- Learning rate too high. A typical starting value is `1e-3` or lower.
- Data not normalized. Inputs should be scaled to a reasonable range.
- Numerical instability in the loss function, for example `log(0)`.
- NaN in the input data before the first forward pass.

Start by lowering the learning rate by a factor of 10 and running again.
""",
    "log_config_mismatch": """
The log and the run config come from different runs. Make sure both files
are from the same training job. If you ran training multiple times, the
config from a previous run will not match the log from the latest one.
""",
    "bob_credentials": """
**Local run:**

1. Create `.streamlit/secrets.toml` in the project root.
2. Add the line: `BOBSHELL_API_KEY = "your_key_here"`.
3. Restart the app.

**Streamlit Cloud:**

1. Open the app settings on Streamlit Cloud.
2. Go to Secrets.
3. Paste the same `BOBSHELL_API_KEY = "your_key_here"` line.
4. Save. The app will reload automatically.

**Where to get a key:** open your Bob workspace settings and generate
an API key.
""",
    "bob_unreachable": """
This is usually a temporary network problem. Click Try again. If it keeps
failing, the Bob service may be down. Check the Bob status page or ask in
the support channel.
""",
    "bob_task": """
Bob ran but did not produce the expected output files. The most common
reasons:

- The task timed out before finishing.
- The workspace was missing the required input files.
- Bob decided the input was not something it could analyze.

Try again. If the problem repeats, check the detail block for Bob's own
error message.
""",
    "bob_timeout": """
Bob can take several minutes on large repositories. Options:

1. Wait longer.
2. Point to a smaller scope, for example a subdirectory of the repository.
3. Try again later when the service is less busy.
""",
    "bobcoins": """
Two options:

1. Top up the balance. Open your Bob workspace settings and add Bobcoins.
2. Use the manual upload path. Run Bob yourself and upload the reports.
   The app will render them without calling Bob.

If you are on a team account, ask your workspace administrator to top up
the shared balance.
""",
    "rate_limit": """
Wait a minute and try again. A rate limit is a temporary block that
clears on its own.
""",
}


def render_error(exc: UserFacingError, retry_key: Optional[str] = None) -> None:
    """Render a UserFacingError as a Streamlit error with a help expander.

    `retry_key` is an optional Streamlit button key. If provided, a
    'Try again' button is rendered next to the message.
    """
    import streamlit as st

    st.error(f"**{exc.headline}**\n\n{exc.explanation}")

    help_text = HELP_BLOCKS.get(exc.help_key)
    if help_text:
        with st.expander("How to fix this"):
            st.markdown(help_text)

    if exc.detail:
        with st.expander("Technical details"):
            st.code(exc.detail, language="text")

    if retry_key is not None:
        if st.button("Try again", key=retry_key):
            st.rerun()


def handle(exc: Exception, retry_key: Optional[str] = None) -> None:
    """Top-level handler. Maps any exception to a user-facing message.

    Use this in every try/except around Bob calls. Raw exceptions never
    reach the user.
    """
    import streamlit as st

    if isinstance(exc, UserFacingError):
        render_error(exc, retry_key=retry_key)
        return

    # Unknown exception - show a generic message, log the traceback
    st.error(
        "**An unexpected error occurred.**\n\n"
        "Please check your files and try again. "
        "If this keeps happening, open the details below."
    )
    with st.expander("Technical details"):
        import traceback
        st.code(traceback.format_exc(), language="text")