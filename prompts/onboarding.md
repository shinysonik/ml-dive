# ML-Dive: Onboarding prompt for IBM Bob 2.0

## Context

Target repo: https://github.com/shinysonik/detectron2-ml-dive (fork of facebookresearch/detectron2)
Goal: produce a structured onboarding report for a new ML engineer joining the project.

## Scenario

A new CV engineer has just been handed this repo. She knows PyTorch, but has never seen Detectron2 before. She needs to understand, in under 30 minutes:

- what the project does and which components matter
- where datasets live and how they are loaded
- which pretrained weights are available and how to load them
- how to run training and inference end-to-end
- which files she should read first, and which she can ignore

## Steps

### Step 1 — Plan mode: scan the repo

Switch Bob to **Plan** mode. Ask Bob to:

1. List top-level directories and files, with one-line descriptions.
2. Identify the main package (`detectron2/`) and its submodules.
3. Point out which submodule is responsible for: config system, model zoo, data loading, training loop, inference.
4. Produce a short "read this first" list of 5 files.

Do not read file contents yet — just the tree and file names.

### Step 2 — Document understanding: read the key files

Still in Plan mode. Ask Bob to open and summarise, one by one:

- `README.md`
- `setup.py`
- `detectron2/config/defaults.py`
- `detectron2/model_zoo/model_zoo.py`
- `detectron2/data/build.py`
- `tools/train_net.py`
- `tools/plain_train_net.py`

For each file, produce:
- purpose in one sentence
- what it depends on
- what depends on it (if visible from imports)
- 3–5 key functions / classes

### Step 3 — Answer the four onboarding questions

Now use **Ask** mode with the knowledge gathered. Ask Bob to answer explicitly:

1. **Where is the data?** Which config keys point to datasets, what format is expected (COCO / LVIS / custom), where is the loader registered.
2. **Where are the weights?** How does `model_zoo.get()` work, what config keys map to checkpoint URLs, where does it cache files locally.
3. **How do I run inference?** Minimal end-to-end example: config + model + image → predictions. Which file to look at.
4. **How do I run training?** Minimal command for `tools/train_net.py`, what config file to start from, how to override dataset paths without editing code.

### Step 4 — Subagent: cross-check

Spawn a **subagent** in Bob focused only on the `config/` directory. Its job:

- list every config node relevant to datasets, checkpoints, and training
- produce a one-page cheat sheet: `key → meaning → example value`

Then compare its output with the answer from Step 3. Flag any contradictions.

### Step 5 — Produce the onboarding report

Ask Bob to output a single markdown file `ONBOARDING_REPORT.md` containing:

- repo overview (5 lines max)
- repo structure diagram (text tree with annotations)
- "read these first" list (5 files, 1 line each)
- datasets section (where, what format, how to point to your own)
- weights section (model zoo, caching, loading)
- inference snippet (working code)
- training command (working shell command)
- config cheat sheet (from Step 4)
- open questions / things Bob was not sure about

## Output format

- Markdown.
- Code blocks for any shell command or Python snippet.
- No emojis, no motivational filler.
- If Bob is uncertain about something, write "unclear" instead of guessing.

## Success criteria

A new engineer can follow the report and run inference + training on a toy dataset without opening any file other than the ones listed in "read these first".