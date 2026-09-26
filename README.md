# ML-Dive

> AI assistant powered by **IBM Bob 2.0** for onboarding into computer-vision repos and debugging training logs.

**Built for the [IBM Bob 2.0 Hackathon](https://lablab.ai) on lablab.ai (September 25–27, 2026).**

---

## The problem

ML engineers lose **days** onboarding into unfamiliar CV repositories and **hours** reading training logs to spot anomalies like NaN loss, overfitting, or loss spikes.

## The solution

**ML-Dive** is an AI assistant that does two things:

1. **Onboarding** — parses a CV repository, explains its structure, locates data and weights, and shows how to run training and inference.
2. **Log debugging** — analyzes training logs, detects anomalies (NaN loss, overfitting, loss spikes, stalled LR), and suggests concrete fixes with file and parameter references.

Both scenarios share one interface, one assistant, and one workflow — not two disconnected tools.

## How it uses IBM Bob 2.0

- **Agent mode** — end-to-end repo and log analysis.
- **Subagents** — one subagent for repository structure, one for log anomaly detection.
- **Document understanding** — reads README, configs, and code comments to explain the project.

## Demo target

**[Detectron2](https://github.com/facebookresearch/detectron2)** (Meta) — an industry-standard CV framework, ideal for demonstrating deep repo comprehension.

Fork used in the demo: [detectron2-ml-dive](https://github.com/shinysonik/detectron2-ml-dive)

## Project status

🚧 In active development during the hackathon (September 25–27, 2026).

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
├── app/                           # placeholder for the packaged demo
├── prompts/                       # Prompts for IBM Bob 2.0 (manual + auto)
├── logs/                          # Synthetic training data + reference detector & scorer
├── docs/                          # anomalies schema, ground truth
└── README.md
```
Data sources and compliance notes: see docs/DATA_SOURCES.md
## Run locally

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run ml_dive.py
```

For **CV Repository Onboarding**, drop `demo_repo.zip` into the same
Structural intake box — it is a 7-file toy project (`Tiny-Detectron`) built
for this demo, so a fresh clone can exercise that mode too.

For **ML Training Log Debugger**, drop `logs/train_log.csv` +
`logs/detectron2_run.log` into the **Structural intake** box, and attach
`logs/reference_anomalies.json` as the diagnosis file.

> **Note on demo data:** three files ship with the repo on purpose —
> `logs/train_log.csv`, `logs/detectron2_run.log` and
> `logs/reference_anomalies.json` — so a fresh clone can be demoed without
> regenerating anything. (`.gitignore` still blocks everything else matching
> `logs/*.log|*.csv|*.json`.)
>
> To rebuild them anyway:
>
> - `python logs/generate_logs.py` → `logs/train_log.csv`, `logs/run_config.yaml`,
>   `docs/ground_truth_debugging.json`
> - `python logs/reference_detectors.py --json` → `logs/reference_anomalies.json`
> - `logs/detectron2_run.log` — hand-maintained, **no script produces it**
>
> None of this is needed at runtime: the app is *upload-only* and never reads
> from disk.

## Credentials

ML-Dive's "Run with Bob" button calls Bob Shell as a subprocess and requires a Bob API key. This feature only works **locally** — it is automatically hidden on Streamlit Cloud (see note below).

### Local setup

1. Create an API key at [bob.ibm.com](https://bob.ibm.com) → API keys → New key → **Scope: General**.
2. Set it in your terminal before running the app:

```powershell
$env:BOBSHELL_API_KEY = "your-api-key-here"
streamlit run ml_dive.py
```

Or copy [`streamlit_cloud_setup.toml`](streamlit_cloud_setup.toml) to `.streamlit/secrets.toml` and fill in your key — Streamlit will load it automatically.

### Streamlit Cloud

The Bob button is hidden on Cloud because Bob Shell cannot be installed there and the analysis takes up to 15 minutes (well beyond Cloud's ~60s connection timeout). No key is needed for the Cloud deployment — judges use the manual upload workflow instead.

---

## Deploy (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Create an app at [share.streamlit.io](https://share.streamlit.io) pointing at
   this repo and branch.
3. **Set the file path to `ml_dive.py`.** Streamlit's default is
   `streamlit_app.py`, which this project does not use — leaving the default
   makes the deploy "succeed" with no app to show.

`requirements.txt` must list every package imported directly by `ml_dive.py`
(`streamlit`, `pandas`, `numpy`, `altair`). Cloud installs from that file
alone, so an undeclared import breaks the deploy rather than the local run.

## Tests

`tests/` boots the real app headlessly (Streamlit's `AppTest`), uploads the
actual demo files, and asserts specific behaviour. **Run it before committing:**

```powershell
venv\Scripts\python tests\run_all.py
```

133 checks across five scripts: log ↔ anomaly cross-referencing by position,
the non-finite audit total, text-log/CSV agreement, the diagnosis upload flow,
every `fix_status` badge state, and every fix listed above. Exit code is `0`
only when all five pass.

## Team

- **ML / CV / IBM Bob integration** — [@shinysonik](https://github.com/shinysonik)
- **Frontend / Full-stack** — *looking for a teammate*
- **Pitch / Video** — *looking for a teammate*

## License

MIT 
