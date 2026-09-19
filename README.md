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
├── app/          # Streamlit demo
├── prompts/      # Prompts for IBM Bob 2.0
├── logs/         # Synthetic training logs with anomalies
├── docs/         # Notes, screenshots, drafts
└── README.md
```
## Team

- **ML / CV / IBM Bob integration** — [@shinysonik](https://github.com/shinysonik)
- **Frontend / Full-stack** — *looking for a teammate*
- **Pitch / Video** — *looking for a teammate*

## License

MIT 