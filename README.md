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
├── ml_dive.py         # Streamlit app — both modes, single file
├── requirements.txt   # runtime deps (installed by Streamlit Cloud)
├── .streamlit/        # dark theme + config
├── tests/             # verification suite — run before committing
├── app/               # placeholder for the packaged demo
├── prompts/           # Prompts for IBM Bob 2.0
├── logs/              # Synthetic training data + reference detector & scorer
├── docs/              # anomalies schema, ground truth
└── README.md
```

Data sources and compliance notes: see `docs/DATA_SOURCES.md`

## Run locally

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run ml_dive.py
```

For **CV Repository Onboarding**, drop `demo_repo.zip` into the same Structural intake box — it is a 7-file toy project (`Tiny-Detectron`) built for this demo, so a fresh clone can exercise that mode too.

For **ML Training Log Debugger**, drop `logs/train_log.csv` + `logs/detectron2_run.log` into the **Structural intake** box, and attach `logs/reference_anomalies.json` as the diagnosis file.

> **Note on demo data:** three files ship with the repo on purpose — `logs/train_log.csv`, `logs/detectron2_run.log` and `logs/reference_anomalies.json` — so a fresh clone can be demoed without regenerating anything. (`.gitignore` still blocks everything else matching `logs/*.log|*.csv|*.json`.)
>
> To rebuild them anyway:
>
> - `python logs/generate_logs.py` → `logs/train_log.csv`, `logs/run_config.yaml`, `docs/ground_truth_debugging.json`
> - `python logs/reference_detectors.py --json` → `logs/reference_anomalies.json`
> - `logs/detectron2_run.log` — hand-maintained, **no script produces it**
>
> None of this is needed at runtime: the app is *upload-only* and never reads from disk.

## Deploy (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Create an app at [share.streamlit.io](https://share.streamlit.io) pointing at this repo and branch.
3. **Set the file path to `ml_dive.py`.** Streamlit's default is `streamlit_app.py`, which this project does not use — leaving the default makes the deploy "succeed" with no app to show.

`requirements.txt` must list every package imported directly by `ml_dive.py` (`streamlit`, `pandas`, `numpy`, `altair`). Cloud installs from that file alone, so an undeclared import breaks the deploy rather than the local run.

## Tests

`tests/` boots the real app headlessly (Streamlit's `AppTest`), uploads the actual demo files, and asserts specific behaviour. **Run it before committing:**

```powershell
venv\Scripts\python tests\run_all.py
```

133 checks across five scripts: log ↔ anomaly cross-referencing by position, the non-finite audit total, text-log/CSV agreement, the diagnosis upload flow, every `fix_status` badge state, and every fix listed above. Exit code is `0` only when all five pass.

## Security

This repository uses security files provided by the IBM Hackathon template:

- **`.gitignore`** — prevents committing credentials and live session files
- **`.bobignore`** — prevents AI assistants from logging credentials
- **`.env.example`** — template for environment variables
- **`SECURITY.MD`** — detailed security guidelines

Before every commit, verify that no credentials are staged:

```powershell
git diff --staged
git check-ignore -v .env
```

## Team

- **ML / CV / prompt engineering** — [@shinysonik](https://github.com/shinysonik)
- **Frontend / Full-stack** — [@Crysnow](https://github.com/Crysnow)
- **Backend / IBM Bob integration** — [@abdullahxyz85](https://github.com/abdullahxyz85)

## License

MIT
