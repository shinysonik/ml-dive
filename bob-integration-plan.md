# ML-Dive × Bob Shell Integration Plan

## Status: PENDING

---

## Top-Level Overview

**Goal:** Connect IBM Bob Shell to the ML-Dive Streamlit app so that a user can provide a GitHub repository URL and a training log CSV, click a single button, and receive the three artifacts (ONBOARDING_REPORT.md, TRIAGE_REPORT.md, anomalies.json) rendered inline — with no manual file uploads required.

**Scope:**
- A new `bob_runner.py` module containing the two public functions `run_bob_onboarding` and `run_bob_triage`.
- A new "Run with Bob" UI section in `ml_dive.py` for each mode (onboarding and debugger).
- Session-state wiring so Bob's output feeds the existing renderers with zero changes to render code.
- Error handling for every failure the user can encounter (network, auth, bad URL, bad log, Bob timeout).
- Credentials loaded from `st.secrets` on Streamlit Cloud and from environment variables locally.
- Manual upload as the existing fallback — no upload code is removed.

**Non-goals:**
- Changing any rendering or charting code.
- Adding new anomaly types or detectors.
- Changing the anomalies schema.

---

## How Bob Shell Works (established facts)

Bob Shell (`bob`) is a CLI tool invoked as a subprocess:
```
bob --auth-method api-key -p "prompt text" --yolo
```
Key facts that shape every design decision below:

1. **CLI, not HTTP/SDK.** No Python SDK exists. Integration is via `subprocess.run`.
2. **Needs a local directory as workspace.** Bob cannot clone a repo from a URL itself. We must `git clone` the repo into a temp folder first, then run Bob from that folder.
3. **Synchronous.** `bob -p "..."` blocks until done. Exit code 0 = success, non-zero = failure.
4. **`--yolo` required for file writes.** Without it Bob is read-only. Our prompts produce output files on disk, so `--yolo` must be passed.
5. **Outputs are files on disk.** The triage workflow writes `TRIAGE_REPORT.md` and `analysis/anomalies.json` into the workspace. The onboarding workflow writes `ONBOARDING_REPORT.md`. We read them back after the subprocess exits.
6. **Credential env var: `BOBSHELL_API_KEY`.** Must be set before the subprocess is launched.
7. **Prompts are long.** Our prompts (in `prompts/debugging.md` and `prompts/onboarding.md`) are multi-step. The non-interactive `-p` flag accepts a single prompt string. We pipe the prompt text via stdin using `cat prompt.txt | bob --yolo`.
8. **Working directory matters.** Bob only writes files inside the directory where it was started. We use a `tempfile.TemporaryDirectory` as the workspace.

---

## Sub-Tasks

---

### Sub-Task 1 — Create `bob_runner.py`: the Bob Shell integration module

**Status:** [ ] pending

**Intent:**
Isolate all Bob Shell interaction in one file. `ml_dive.py` imports two functions from it and never touches subprocess, tempfile, or git directly. This keeps the app clean and makes the integration easy to test in isolation.

**Expected Outcomes:**
- `bob_runner.py` exists alongside `ml_dive.py` in the project root.
- `run_bob_onboarding(repo_url: str) -> str` clones the repo, runs Bob's onboarding prompt, reads back `ONBOARDING_REPORT.md`, and returns its content as a string.
- `run_bob_triage(log_csv_bytes: bytes, run_config_bytes: bytes) -> tuple[str, dict]` copies the log and config into a temp workspace, runs Bob's triage prompt, reads back `TRIAGE_REPORT.md` and `analysis/anomalies.json`, and returns `(triage_md_str, anomalies_dict)`.
- Credentials are read from the environment (`BOBSHELL_API_KEY`); if missing a `BobCredentialError` is raised immediately (before any subprocess is launched).
- All five error conditions listed in the brief raise a specific named exception: `BobCredentialError`, `BobUnreachableError`, `BobTaskError`, `InvalidRepoError`, `MalformedLogError`.
- A `get_bob_api_key()` helper reads from `st.secrets["BOBSHELL_API_KEY"]` if Streamlit context is active, otherwise from `os.environ["BOBSHELL_API_KEY"]`.

**Todo List:**
1. Create `bob_runner.py` with the five exception classes.
2. Implement `get_bob_api_key()` — try `st.secrets`, fall back to `os.environ`, raise `BobCredentialError` if neither is set.
3. Implement `_run_bob_in_dir(workspace: Path, prompt: str, timeout_seconds: int)` — the shared subprocess helper that sets `BOBSHELL_API_KEY` in the subprocess environment, writes the prompt to a temp file, pipes it with `cat prompt.txt | bob --auth-method api-key --yolo`, captures stdout+stderr, raises `BobUnreachableError` on `FileNotFoundError` (bob not installed), raises `BobTaskError` on non-zero exit code (attaching the stderr tail for debugging).
4. Implement `run_bob_onboarding(repo_url: str) -> str`:
   a. Validate that `repo_url` looks like a git URL (simple regex: starts with `https://` or `git@`); raise `InvalidRepoError` with a clear message if not.
   b. Create a `tempfile.TemporaryDirectory`.
   c. Run `git clone --depth 1 repo_url workspace_dir` via subprocess; on non-zero exit raise `InvalidRepoError` (repo private, does not exist, or network failure).
   d. Copy `prompts/onboarding.md` into the workspace as `bob_prompt.txt`.
   e. Call `_run_bob_in_dir(workspace, prompt, timeout=600)`.
   f. Read `ONBOARDING_REPORT.md` from the workspace root; if absent raise `BobTaskError("Bob ran but did not produce ONBOARDING_REPORT.md")`.
   g. Return the file content as a string.
5. Implement `run_bob_triage(log_csv_bytes: bytes, run_config_bytes: bytes) -> tuple[str, dict]`:
   a. Validate `log_csv_bytes` — parse the first line with `csv.reader`; if it raises or the first header row does not contain `iteration` raise `MalformedLogError`.
   b. Create a `tempfile.TemporaryDirectory`.
   c. Write `logs/train_log.csv` and `logs/run_config.yaml` into the workspace.
   d. Copy `prompts/debugging.md` into the workspace as `bob_prompt.txt`.
   e. Call `_run_bob_in_dir(workspace, prompt, timeout=900)`.
   f. Read `TRIAGE_REPORT.md` from the workspace root; if absent raise `BobTaskError`.
   g. Read `analysis/anomalies.json`; if absent or not valid JSON raise `BobTaskError`.
   h. Return `(triage_md_str, anomalies_dict)`.

**Relevant Context:**
- Prompt files: `prompts/onboarding.md`, `prompts/debugging.md`
- Bob Shell docs: auth via `BOBSHELL_API_KEY`, invoked as `bob --auth-method api-key -p "..." --yolo`
- Bob writes files inside its working directory only
- Streamlit secrets: `st.secrets["BOBSHELL_API_KEY"]` on Cloud; `os.environ` locally

---

### Sub-Task 2 — Add "Run with Bob" UI to Repository Onboarding mode

**Status:** [ ] pending

**Intent:**
Give the user a one-click path to get the onboarding report without any manual upload. The Bob section appears above the existing manual upload fallback. If Bob succeeds, the report renders automatically. If Bob fails, the manual upload is still available and a clear error message explains what went wrong.

**Expected Outcomes:**
- In the onboarding panel, above the existing `st.file_uploader`, there is a bordered container labeled "⚡ Run Bob Onboarding".
- The container has a text input for the GitHub URL and a "Run Bob" button.
- When clicked, a spinner runs (`st.spinner("Bob is analyzing the repository…")`).
- On success: the report content is stored in `st.session_state["ob_bob_report"]` and rendered without any upload widget interaction.
- On each of the five error types, a specific `st.error(...)` message is shown (no Python traceback, no exception text). The error text will be written by the teammate as specified.
- The existing `st.file_uploader` widget for `ONBOARDING_REPORT.md` remains unchanged below the Bob section, labeled "Or upload manually".
- If both a Bob result and an uploaded file are present, the Bob result takes precedence and a note ("Source: Bob") is shown.

**Todo List:**
1. In `render_onboarding()` in `ml_dive.py`, before the existing `st.file_uploader` block (around line 959), insert the "Run with Bob" container.
2. Add a text input (`st.text_input`, key `"ob_repo_url"`) and a button (`st.button("Run Bob", key="ob_run_bob_btn")`).
3. On button click, call `run_bob_onboarding(repo_url)` inside a `try/except` block catching each named exception. Map each to a specific `st.error()` call with a user-visible message (placeholder text, teammate fills in the final wording).
4. On success, write to `st.session_state["ob_bob_report"]` and `st.rerun()`.
5. After the Bob block, modify the existing report resolution logic: check `st.session_state.get("ob_bob_report")` first, then the file uploader result. Set `report_source` accordingly ("Source: Bob" vs "Uploaded: filename").
6. Add `from bob_runner import run_bob_onboarding, BobCredentialError, BobUnreachableError, BobTaskError, InvalidRepoError, MalformedLogError` to imports at top of `ml_dive.py`.

**Relevant Context:**
- Target function: `render_onboarding()`, lines 907–981 in `ml_dive.py`
- Session key to add: `"ob_bob_report"` (str — the onboarding report markdown)
- Existing session keys to keep: `"ob_code_pick"`, `"ob_report_upload"`

---

### Sub-Task 3 — Add "Run with Bob" UI to ML Training Log Debugger mode

**Status:** [ ] pending

**Intent:**
Give the user a one-click path to get both the triage report and the anomaly diagnosis from Bob, using the log and config already in the structural intake. The Bob section appears above the existing manual uploaders. On success, all three artifacts (zones, triage report) are injected into session state and the existing tabs render them automatically.

**Expected Outcomes:**
- In `render_debugger()`, before the tabs open, there is a bordered container labeled "⚡ Run Bob Triage".
- The container checks `intake.data` (CSV files) and uses the first CSV as the log; if none is present a clear `st.info` message says to upload a CSV first.
- The container also checks for a config file in `intake.code` (YAML files); if none is found it shows a warning but still allows running (config is optional for triage).
- A "Run Bob" button, when clicked, calls `run_bob_triage(log_csv_bytes, run_config_bytes)`.
- On success: `st.session_state["dbg_bob_triage"]` holds the triage report string, and `st.session_state["dbg_bob_zones"]` holds the anomalies dict.
- On each of the five error types, a specific `st.error(...)` is shown with no traceback.
- The existing manual uploaders inside the tabs remain unchanged.
- `render_debugger()` reads `st.session_state.get("dbg_bob_zones")` and uses it as `shared_zones` when the DIAG_UPLOAD_KEY session state value is None.
- `_render_triage_report_tab()` reads `st.session_state.get("dbg_bob_triage")` and renders it if present, before falling back to the file uploader.

**Todo List:**
1. In `render_debugger()` (around line 1240), before the `shared_zones` computation block, insert the "Run with Bob" container.
2. Resolve `log_csv_bytes` from the first item in `intake.data`; resolve `run_config_bytes` from the first `.yaml` file in `intake.code` if present, else `b""`.
3. Add a "Run Bob" button; on click call `run_bob_triage` inside `try/except`, storing results in session state on success.
4. Modify `shared_zones` resolution: if `DIAG_UPLOAD_KEY` is not in session state or is None, fall back to `st.session_state.get("dbg_bob_zones")`.
5. Modify `_render_triage_report_tab()`: check `st.session_state.get("dbg_bob_triage")` before the file uploader. If present, render it (with "Source: Bob" caption) and show the uploader below labeled "Or upload a different report".

**Relevant Context:**
- Target function: `render_debugger()`, lines 1228–1293; `_render_triage_report_tab()`, lines 1297–1320
- `shared_zones` computation block: lines 1253–1264
- Session keys to add: `"dbg_bob_triage"` (str), `"dbg_bob_zones"` (list[dict])
- Existing session keys to keep: `DIAG_UPLOAD_KEY = "dbg_diag"`, `"dbg_triage_upload"`, `"dbg_log_pick"`, `"dbg_csv_pick"`

---

### Sub-Task 4 — Secrets and credentials management

**Status:** [ ] pending

**Intent:**
Ensure the app works on both local machines (env var) and Streamlit Cloud (st.secrets) without hardcoding any credentials, and provide the team with clear instructions for configuring both environments.

**Expected Outcomes:**
- `bob_runner.get_bob_api_key()` transparently handles both sources.
- A `.streamlit/secrets.toml.example` file documents the required secret key for the team.
- `README.md` has a short "Credentials" section explaining: set `BOBSHELL_API_KEY` locally, add it to Streamlit Cloud secrets under the same key name.
- `requirements.txt` is updated with any new dependency (only `gitpython` if used; otherwise `subprocess` is stdlib).

**Todo List:**
1. Confirm that `git clone` is done via `subprocess` (stdlib, no new dependency) rather than `gitpython`.
2. Create `.streamlit/secrets.toml.example` with the placeholder `BOBSHELL_API_KEY = "your-key-here"` and a comment explaining it.
3. Add a "Credentials" section to `README.md`: local env var, Streamlit Cloud secrets panel.
4. Verify `requirements.txt` needs no new entries (subprocess, tempfile, csv, os are all stdlib).

**Relevant Context:**
- Existing `.streamlit/config.toml` already exists (dark theme)
- `get_bob_api_key()` implementation is in Sub-Task 1

---

### Sub-Task 5 — Error handling and user-visible messages

**Status:** [ ] pending

**Intent:**
Replace every possible Python traceback with a specific, friendly message. The user should never see an exception. Each of the five failure modes has a distinct message so the user knows exactly what went wrong and what to try next.

**Expected Outcomes:**
- `BobCredentialError` → "Bob API key not found. Add BOBSHELL_API_KEY to your environment variables (local) or Streamlit secrets (Cloud)."
- `BobUnreachableError` → "Bob Shell is not installed or not on PATH. Install it from the Bob portal and ensure the `bob` command is available."
- `InvalidRepoError` → "Could not clone the repository. Check that the URL is correct and the repository is public." (+ any detail from git's stderr)
- `MalformedLogError` → "The training log file is empty or missing the `iteration` column. Check that it is a valid CSV with at least: iteration, train_loss, val_loss, lr."
- `BobTaskError` → "Bob ran but did not produce the expected output files. This may mean the task timed out or Bob's response was incomplete. Try again." (+ the last 20 lines of Bob's stderr, shown in an expander)
- Generic `Exception` → "An unexpected error occurred. Please check your files and try again." (+ expander with traceback for developer debugging)
- A "Run Bob again" button is shown after any error so the user does not have to reload.

**Todo List:**
1. Define message constants in `bob_runner.py` (one per error class) so `ml_dive.py` imports them.
2. In both Bob UI sections (Sub-Tasks 2 and 3), implement the full `try/except` chain: each named exception → `st.error(MESSAGE)`, generic `Exception` → `st.error(GENERIC_MSG)` + `st.expander("Details")` with traceback.
3. In `BobTaskError`, store Bob's stderr tail in the exception so `ml_dive.py` can display it in an expander.
4. Ensure no `st.exception()` or unguarded `raise` reaches the user.

**Relevant Context:**
- Exception classes defined in Sub-Task 1
- `_run_bob_in_dir` captures stderr and attaches the tail to `BobTaskError`

---

### Sub-Task 6 — End-to-end smoke test and deployment checklist

**Status:** [ ] pending

**Intent:**
Verify the full flow works — both the Bob path and the manual fallback — before the demo. Document what must be confirmed on Streamlit Cloud specifically.

**Expected Outcomes:**
- A `bob_runner_test.py` script that can be run locally to confirm: credentials are loadable, `run_bob_onboarding` returns a non-empty string given a real public repo URL, `run_bob_triage` returns a tuple with a non-empty string and a non-empty dict given the sample `logs/train_log.csv` and `logs/run_config.yaml`.
- A deployment checklist in `DEPLOY_CHECKLIST.md`: install Bob Shell on the server, set `BOBSHELL_API_KEY` in Streamlit Cloud secrets, accept the license agreement, trust the workspace, confirm `git` is available on the server PATH.
- All five error paths confirmed to show clean messages (test by temporarily passing a bad URL and a malformed CSV).

**Todo List:**
1. Write `bob_runner_test.py` with three test functions: `test_credentials()`, `test_onboarding()`, `test_triage()`. Each prints PASS or FAIL with a reason. Not a pytest file — just a plain `if __name__ == "__main__"` runner.
2. Write `DEPLOY_CHECKLIST.md` covering: Bob Shell installation on Streamlit Cloud (it needs to be a package or CLI available on PATH — note this as an open question to verify with the team), secrets setup, license acceptance, and a "before demo" verification step.
3. Note the open question: Streamlit Cloud is a managed environment; confirm whether `bob` (Bob Shell) can be installed as a system dependency. If not, the live Bob path will only work locally and the Cloud deployment will use the manual upload fallback. The plan is valid either way — document both scenarios.

**Relevant Context:**
- Sample files: `logs/train_log.csv`, `logs/run_config.yaml`
- Reference output: `logs/reference_anomalies.json`
- Existing tests: `tests/run_all.py` (133 checks) — these should still pass after the integration

---

## Architecture Diagram (reference)

```
User browser
    │
    ▼
Streamlit app (ml_dive.py)
    │
    ├── "Run with Bob" button (onboarding mode)
    │       │
    │       └── bob_runner.run_bob_onboarding(repo_url)
    │               │
    │               ├── git clone → /tmp/workspace/
    │               ├── bob --auth-method api-key --yolo  [subprocess, blocking]
    │               └── read /tmp/workspace/ONBOARDING_REPORT.md → return str
    │
    ├── "Run with Bob" button (debugger mode)
    │       │
    │       └── bob_runner.run_bob_triage(log_bytes, config_bytes)
    │               │
    │               ├── write log + config → /tmp/workspace/logs/
    │               ├── bob --auth-method api-key --yolo  [subprocess, blocking]
    │               ├── read /tmp/workspace/TRIAGE_REPORT.md → str
    │               └── read /tmp/workspace/analysis/anomalies.json → dict
    │
    └── st.session_state (shared)
            ├── ob_bob_report     → feeds render_onboarding render path
            ├── dbg_bob_triage    → feeds _render_triage_report_tab
            └── dbg_bob_zones     → feeds shared_zones in render_debugger
```

---

## Open Questions (resolve before implementation)

1. **Streamlit Cloud + Bob Shell installation:** Streamlit Cloud does not allow arbitrary system packages by default. Confirm whether `bob` (Bob Shell) can be installed via a `packages.txt` or `Dockerfile` override, or whether the live Bob path is only available in local/server deployments. If it cannot be installed on Cloud, the app on Cloud will display a clear message ("Bob Shell is not available in this environment — use the manual upload.") and fall back gracefully.

2. **Bob prompt style for non-interactive mode:** The existing prompts in `prompts/debugging.md` and `prompts/onboarding.md` are multi-step operator guides, not single prompts. For the non-interactive `-p` flag we need to either: (a) pass the full document as one prompt, or (b) distill it into a single consolidated prompt per workflow. Decide which approach before Sub-Task 1 implementation.

3. **Timeout values:** The triage workflow is set to 900 seconds (15 min) and onboarding to 600 seconds (10 min). Confirm these are acceptable for the demo environment. Streamlit Cloud will kill a request after ~60s of inactivity — if Bob runs as a background task this needs to be addressed.

4. **`git` on Streamlit Cloud PATH:** Confirm `git` is available by default on Streamlit Cloud (it is on most managed Python environments, but worth checking before the demo).
