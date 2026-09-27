# ML-Dive Deployment Checklist

> Use this checklist before every demo and before every production deployment.
> Mark each item ✅ as you confirm it.  The checklist covers two scenarios:
> **local/server deployment** (Bob path available) and
> **Streamlit Community Cloud deployment** (manual-upload fallback only).

---

## Scenario A — Local / Server Deployment (Bob path active)

### 1. Bob Shell installation

- [ ] Bob Shell CLI is installed:
  ```powershell
  powershell -c "irm https://bob.ibm.com/download/bobshell.ps1 | iex"
  ```
- [ ] The `bob` command is on PATH and returns a version:
  ```powershell
  bob --version
  ```
- [ ] On Windows, `bob.cmd` is the actual executable name — `bob_runner.py`
  uses `bob.cmd` on `os.name == "nt"` automatically.  No manual fix needed.

### 2. Bob API key

- [ ] You have a valid Bob API key (create one at
  [bob.ibm.com](https://bob.ibm.com) → API keys → New key → **Scope: General**).
- [ ] The key is set in the environment **before** starting the app:
  ```powershell
  $env:BOBSHELL_API_KEY = "your_api_key_here_DO_NOT_COMMIT"
  streamlit run ml_dive.py
  ```
  *Alternatively*, copy [`streamlit_cloud_setup.toml`](streamlit_cloud_setup.toml)
  to `.streamlit/secrets.toml` and fill in the key — Streamlit loads it
  automatically and `get_bob_api_key()` finds it there first.
- [ ] The key is **not** committed to git (check `.gitignore` includes
  `.streamlit/secrets.toml` and `.env`).

### 3. License acceptance

- [ ] Run Bob interactively once to accept the EULA before the demo:
  ```powershell
  bob --version
  ```
  If the license prompt appears, type `yes` and press Enter.  Subsequent
  subprocess calls (with `--accept-license`) will not re-prompt.

### 4. Git availability

- [ ] `git` is available on PATH (required by `run_bob_onboarding` for shallow
  clone):
  ```powershell
  git --version
  ```
- [ ] The machine can reach `github.com` on port 443 (public clone over HTTPS).

### 5. Prompt files

- [ ] `prompts/onboarding_auto.md` exists (used by `run_bob_onboarding`).
- [ ] `prompts/debugging_auto.md` exists (used by `run_bob_triage`).

### 6. Smoke test — run before the demo

Run the layered smoke test from the project root:

```powershell
# Layer 1 + 2 (no Bob calls, fast):
python bob_runner_test.py

# Full end-to-end (requires key + bob installed, takes several minutes):
python bob_runner_test.py --e2e
```

Expected output: all tests print `[PASS]`, exit code `0`.

Also confirm the five error paths display clean messages (no Python traceback):

| Scenario | How to trigger | Expected UI message |
|---|---|---|
| Missing key | Unset `BOBSHELL_API_KEY` before clicking Run | `BobCredentialError` message |
| Bob not on PATH | Rename `bob.cmd` temporarily | `BobUnreachableError` message |
| Bad repo URL | Enter `ftp://not-a-repo` in the URL field | `InvalidRepoError` message |
| Private/missing repo | Enter `https://github.com/does-not-exist-xyz/repo` | `InvalidRepoError` message |
| Malformed CSV | Upload a CSV without an `iteration` column | `MalformedLogError` message |

### 7. Existing test suite

Confirm the 133 existing checks still pass:

```powershell
venv\Scripts\python tests\run_all.py
```

Exit code must be `0`.

---

## Scenario B — Streamlit Community Cloud (manual-upload fallback)

### ⚠️ Open Question — Bob Shell on Streamlit Cloud

**Streamlit Community Cloud is a managed environment.  Custom system binaries
(`bob`) cannot be installed there today.**  The app detects this and hides the
"Run with Bob" button automatically (the `BobUnreachableError` path shows a
clear info message instead of an error).  Judges and users on Cloud use the
manual file-upload workflow.

If the team later moves to a self-hosted server or a Docker-based deployment,
follow Scenario A above.

### 1. Repository and app setup

- [ ] The repository is pushed to GitHub (public or private with Cloud access).
- [ ] A Streamlit Cloud app is created at
  [share.streamlit.io](https://share.streamlit.io) pointing at this repo.
- [ ] **Main file path is set to `ml_dive.py`** (Cloud defaults to
  `streamlit_app.py` — leaving the default causes a silent empty deploy).

### 2. Secrets on Cloud

No Bob API key is needed for the Cloud deployment (the Bob button is hidden).
If you ever enable a Cloud path for Bob, add the key in the Streamlit Cloud
secrets panel:

```
# Settings → Secrets → paste the following:
BOBSHELL_API_KEY = "your_api_key_here"
```

### 3. Requirements

- [ ] `requirements.txt` lists all packages imported by `ml_dive.py`:
  `streamlit`, `pandas`, `numpy`, `altair`.
- [ ] No new runtime dependencies were added by the Bob integration (all modules
  used — `subprocess`, `tempfile`, `csv`, `os`, `json` — are Python stdlib).

### 4. `git` on Cloud

`git` is available by default on Streamlit Community Cloud's managed Python
environment.  No `packages.txt` entry is needed.

### 5. Before-demo verification on Cloud

- [ ] Open the deployed app URL and confirm it loads without errors.
- [ ] Upload `logs/train_log.csv` + `logs/reference_anomalies.json` via the
  Structural Intake box and confirm the anomaly view renders correctly.
- [ ] Upload `ONBOARDING_REPORT.md` (from a previous local Bob run or the
  sample in `docs/`) and confirm the report renders correctly.
- [ ] Confirm the "Run with Bob" section is **not visible** (hidden on Cloud).

---

## Quick Reference — Key Files

| File | Purpose |
|---|---|
| `bob_runner.py` | All Bob Shell subprocess logic |
| `bob_runner_test.py` | Layered smoke test (unit / auth / e2e) |
| `prompts/onboarding_auto.md` | Onboarding prompt sent to Bob |
| `prompts/debugging_auto.md` | Triage prompt sent to Bob |
| `streamlit_cloud_setup.toml` | Secrets template — copy to `.streamlit/secrets.toml` |
| `tests/run_all.py` | Full 133-check test suite |
| `logs/train_log.csv` | Sample training log for triage smoke test |
| `logs/run_config.yaml` | Sample run config for triage smoke test (generate with `python logs/generate_logs.py`) |
| `logs/reference_anomalies.json` | Reference anomaly output for manual upload demo |
