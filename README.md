# ML-Dive

> AI assistant powered by **IBM Bob 2.0** for onboarding into computer-vision repos and debugging training logs.

**Built for the [IBM Bob 2.0 Hackathon](https://lablab.ai) on lablab.ai (September 25–27, 2026).**

---

## ⚠️ Important: how to see IBM Bob in action

**The live Streamlit Cloud demo runs without IBM Bob.** Bob Shell cannot be installed on Streamlit Cloud, and a full analysis run takes up to 15 minutes — well beyond Cloud's ~60-second connection timeout. The "Run with Bob" button is automatically hidden there.

**The demo video was recorded running the app locally, specifically so Bob would be visible.** If you run ML-Dive on Streamlit Cloud instead, you will not see Bob at all — that is expected, not a bug. To see the full Bob-powered workflow yourself, run the app locally (see [Run locally](#run-locally-full-bob-experience)).

On Cloud, judges can still use the **manual upload workflow**: drop a repository ZIP or training log, and the app runs its deterministic detector and report generation without Bob.

---

## The problem

ML engineers lose **days** onboarding into unfamiliar CV repositories and **hours** reading training logs to spot anomalies like NaN loss, overfitting, or loss spikes.

> **This isn't just an inconvenience — it's a direct cost.** A NaN-loss run or a silently diverging training job that nobody catches for a few hours keeps burning GPU time on a checkpoint that was never going to be usable. On a multi-GPU cloud instance, that's real, avoidable compute spend — on top of the engineer-hours spent tracking the problem down after the fact.

## The solution

**ML-Dive** is an AI assistant that does two things:

1. **Repository onboarding** — parses a CV repository, explains its structure, locates data and weights, and shows how to run training and inference.
2. **Training log debugging** — analyzes training logs, detects anomalies (NaN loss, overfitting, loss spikes, stalled LR), and suggests concrete fixes with file and parameter references.

Both scenarios share one interface, one assistant, and one workflow — not two disconnected tools.

## How it uses IBM Bob 2.0

- **Agent mode** — end-to-end repo and log analysis.
- **Subagents** — one subagent for repository structure, one for log anomaly detection.
- **Document understanding** — reads README, configs, and code comments to explain the project.

## Demo target

**[Detectron2](https://github.com/facebookresearch/detectron2)** (Meta) — an industry-standard CV framework, ideal for demonstrating deep repo comprehension.

Fork used in the demo: [detectron2-ml-dive](https://github.com/shinysonik/detectron2-ml-dive)

## Project status

🚧 Built during the hackathon (September 25–27, 2026). The local app is fully functional. The Cloud deployment is a read-only showcase without Bob.

---

## Run locally (full Bob experience)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run ml_dive.py
```

**To enable the "Run with Bob" button**, set your Bob API key before starting:

```powershell
$env:BOBSHELL_API_KEY = "your_api_key_here_DO_NOT_COMMIT"
streamlit run ml_dive.py
```

Or copy [`streamlit_cloud_setup.toml`](streamlit_cloud_setup.toml) to `.streamlit/secrets.toml` and fill in your key — Streamlit will load it automatically.

Get a key at [bob.ibm.com](https://bob.ibm.com) → API keys → New key → Scope: General.

For **Repository Onboarding**, drop `demo_repo.zip` (a 7-file toy project, `Tiny-Detectron`) into the **Structural intake** box.

For **Training Log Debugger**, drop `logs/train_log.csv` + `logs/detectron2_run.log` into **Structural intake**, and attach `logs/reference_anomalies.json` as the diagnosis file.

> **Note on demo data:** three files ship with the repo on purpose — `logs/train_log.csv`, `logs/detectron2_run.log`, and `logs/reference_anomalies.json` — so a fresh clone can be demoed without regenerating anything. `logs/reference_anomalies.json` is a synthetic, reproducible reference set (not output from a real Detectron2 training run) — we used it because there wasn't time during the hackathon to run a full real training job and label its anomalies by hand. It exists so the Log Debugger tab has a diagnosis file to attach; in normal use, this file is what Bob itself produces from the raw logs.
>
> To rebuild the demo data anyway:
>
> - `python logs/generate_logs.py` → `logs/train_log.csv`, `logs/run_config.yaml`, `docs/ground_truth_debugging.json`
> - `python logs/reference_detectors.py --json` → `logs/reference_anomalies.json`
> - `logs/detectron2_run.log` — hand-maintained, **no script produces it**
>
> None of this is needed at runtime: the app is *upload-only* and never reads from disk.

## Deploy on Streamlit Community Cloud (without Bob)

1. Push this repository to GitHub.
2. Create an app at [share.streamlit.io](https://share.streamlit.io) pointing at this repo and branch.
3. **Set the file path to `ml_dive.py`.** Streamlit's default is `streamlit_app.py`, which this project does not use — leaving the default makes the deploy "succeed" with no app to show.
4. No API key is needed. Bob Shell is unavailable on Cloud, so the Run button is hidden automatically.

`requirements.txt` must list every package imported directly by `ml_dive.py` (`streamlit`, `pandas`, `numpy`, `altair`). Cloud installs from that file alone, so an undeclared import breaks the deploy rather than the local run.

## Tests

`tests/` boots the real app headlessly (Streamlit's `AppTest`), uploads the actual demo files, and asserts specific behaviour. **Run it before committing:**

```powershell
venv\Scripts\python tests\run_all.py
```

133 checks across five scripts: log ↔ anomaly cross-referencing by position, the non-finite audit total, text-log/CSV agreement, the diagnosis upload flow, every `fix_status` badge state, and every fix listed above. Exit code is `0` only when all five pass.

## Repository structure

```
ml-dive/
├── ml_dive.py                     # Streamlit app — both modes, single file
├── bob_runner.py                  # Bob Shell integration (local only)
├── edge_cases.py                  # error classes + validators + UI helpers
├── requirements.txt               # runtime deps (installed by Streamlit Cloud)
├── streamlit_cloud_setup.toml     # secrets template — copy to .streamlit/secrets.toml
├── .streamlit/                    # dark theme + config
├── tests/                         # verification suite — run before committing
├── prompts/                       # Prompts for IBM Bob 2.0 (manual + auto)
├── logs/                          # Synthetic training data + reference detector & scorer
├── docs/                          # anomalies schema, ground truth
└── README.md
```

Data sources and compliance notes: see `docs/DATA_SOURCES.md`

## Team — ML-Dive

- **Sofia Shevelo** — team lead, CV / ML engineer — [@shinysonik](https://github.com/shinysonik)
- **Abdullah Jameel** — IBM Bob integration — [@abdullahxyz85](https://github.com/abdullahxyz85)
- **Aviral Gupta** — app & frontend, pitch / demo video — [@Crysnow](https://github.com/Crysnow)

Repository: [github.com/shinysonik/ml-dive](https://github.com/shinysonik/ml-dive)

## License

MIT
