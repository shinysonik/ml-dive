# ML-Dive: Onboarding scenario for IBM Bob 2.0

## Scenario

Target repo: `github.com/shinysonik/detectron2-ml-dive` (fork of facebookresearch/detectron2). A new CV engineer has been handed it. They know PyTorch, not Detectron2. In under 30 minutes they need to know:

- what the project does and which components matter
- where datasets live and how they are loaded
- which pretrained weights exist and how they are loaded
- how to run inference and training end to end
- which files to read first and which to ignore

## Preconditions (operator, not for Bob)

- Open Bob in a workspace rooted at the Detectron2 fork.
- Mode names below (Plan, Ask, Agent) follow the hackathon material. Verify they match the IDE. If Plan mode cannot write files, run the file-writing parts of Step 1 in Agent mode.
- Outputs: intermediate files in `onboarding/`, final `ONBOARDING_REPORT.md` in the workspace root. After the run, copy both into `docs/examples/onboarding/` in the ml-dive repo.
- Do not let Bob install anything until Step 5.

## Step 1 - Plan mode: rules, tree, key files

Mode: Plan. Two pastes.

Paste 1, session rules. If Bob supports a project rules file, put this block there instead.

```
Save the following rules verbatim to onboarding/RULES.md, then follow them for the whole session.
- Markdown. Code blocks for shell commands and Python. No emojis, no motivational filler.
- Every factual claim about the repository cites path:line, taken from a file you opened or a grep result in this session. Do not cite from memory.
- Separate "verified in code" from "stated in documentation". If they differ, report the difference.
- Tag a claim "hypothesis — verify in repo" only if it is about code you did not open in this session. Claims from files you opened cite path:line and stand alone, without caveats.
- If something is unclear, write "unclear". Do not guess.
- Do not run pip, conda or any install command unless a step tells you to.
- Every code block in the final report carries a status line: "executed", "static check only", or "hypothesis — verify in repo".
- Write results of each step to onboarding/, one file per step.
```

Paste 2, tree and key files.

```
Part A, directory listings only, no file contents. Write onboarding/01_tree.md with:
1. Top-level files and directories, one line each. Descriptions come from names only and are marked "inferred".
2. The main package detectron2/ and its submodules (one level), one line each.
3. Which submodule is responsible for: config system, model zoo, data loading and dataset registration, training loop, inference, checkpoint loading. Detectron2 may contain more than one config system: list every config-related path you see (configs/, detectron2/config/, projects/, tools/) without concluding yet.
4. Output of git log --oneline -20, and which commits are not upstream, or "none visible".
5. Confirm that each of these files exists in this checkout; if one does not, say so and do not substitute:
   README.md, INSTALL.md, GETTING_STARTED.md, MODEL_ZOO.md, setup.py,
   detectron2/config/defaults.py, detectron2/model_zoo/model_zoo.py,
   detectron2/data/catalog.py, detectron2/data/datasets/builtin.py, detectron2/data/build.py,
   detectron2/engine/defaults.py, detectron2/checkpoint/detection_checkpoint.py, detectron2/utils/file_io.py,
   tools/train_net.py, tools/plain_train_net.py, tools/lazyconfig_train_net.py,
   demo/demo.py, datasets/README.md
   Then list up to 3 more files you think are missing from this reading list, each with a one-line reason.

Part B, after Part A is written. Read the files confirmed in onboarding/01_tree.md, in the order listed, one at a time. Reading limits: MODEL_ZOO.md - first 60 lines and section headings only. detectron2/config/defaults.py - only the sections DATASETS, DATALOADER, INPUT, MODEL.WEIGHTS, MODEL.DEVICE, SOLVER, TEST, OUTPUT_DIR. detectron2/data/datasets/builtin.py - registration functions and how the dataset root is determined.
For each file write to onboarding/02_files.md:
- Purpose: one sentence.
- Depends on: from its import statements, with line numbers.
- Used by: from a grep of imports across the repo; count and 3 examples.
- Key symbols: 3 to 5 functions or classes with path:line.
After the last file, add the section "Read first": the 5 files a new engineer should read first, one line each with the reason, and a list of files they can skip.
```

Checkpoint: `onboarding/RULES.md`, `01_tree.md`, `02_files.md` exist; "Read first" has exactly 5 files; missing files are listed.

## Step 2 - Ask mode: answer the onboarding questions

Mode: Ask. Paste:

```
Answer from onboarding/02_files.md and the repository. Cite path:line for every statement.

Q1 Data. Which config keys select datasets. How a dataset name maps to a loader (registration in the catalog). Which formats are built in. Where the dataset root comes from (environment variable or default path). The minimum code needed to register a custom COCO-format dataset. Whether datasets can be selected or relocated without editing code - answer separately for built-in and for custom datasets.
Q2 Weights. How model_zoo.get() and the checkpoint-URL helper work. How a config file maps to a checkpoint URL. How MODEL.WEIGHTS is resolved to a local file (which path handlers, which class loads it). Where downloaded files are cached locally.
Q3 Inference. A minimal end-to-end example: config + weights + one image -> predictions. Which class does it, expected image format, structure of the output. Cover both config systems if both exist.
Q4 Training. The minimal command for tools/train_net.py with a config from configs/. How command-line overrides are parsed. What extra code is needed to train on a custom dataset besides overrides. What tools/plain_train_net.py and tools/lazyconfig_train_net.py are for and when to use each.
Q5 Install. Steps from INSTALL.md and setup.py, hard requirements, and the two most likely failures.

End with a list titled "Unclear" for everything you could not confirm in files.
```

Checkpoint: Q1 and Q4 distinguish built-in from custom datasets; the "Unclear" list is present.

## Step 3 - Agent mode, subagent: cross-check the config

Mode: Agent. Paste:

```
First save your previous answers verbatim to onboarding/03_answers.md.

Then spawn one subagent. It first reads onboarding/RULES.md, may read the whole repository, and writes only to onboarding/. Its task:
1. List every CfgNode key relevant to datasets, weights and checkpoints, the training loop and solver, and inference. For each: key, meaning (from the code comment or docstring), default value, an example value taken from one file in configs/.
2. For each key, find the file:line that consumes it outside detectron2/config/ (grep for cfg.<KEY>). If none is found, write "no consumer found".
3. A section of at most 10 lines describing how LazyConfig differs from CfgNode. Nothing more.
Write to onboarding/04_config_cheatsheet.md.

Then compare the cheat sheet with onboarding/03_answers.md and write onboarding/04_contradictions.md: each contradiction as (claim, cheat sheet value, file:line for both), or "none found".
```

Fallback if subagents are unavailable. Paste:

```
Subagents are not available. Save your previous answers verbatim to onboarding/03_answers.md. Then do the following yourself, in order: (1) build onboarding/04_config_cheatsheet.md with columns key, meaning, default, example value, consumer file:line; (2) add a section of at most 10 lines on how LazyConfig differs; (3) compare with onboarding/03_answers.md and write onboarding/04_contradictions.md.
```

Checkpoint: every cheat sheet row has a consumer or "no consumer found".

## Step 4 - Agent mode: toy scripts and report

Mode: Agent. Paste:

```
Write these files:
- onboarding/make_toy_dataset.py: generates 4 synthetic 256x256 images, each with one filled rectangle, plus a COCO-format annotation JSON with one category "box". No downloads.
- onboarding/toy_infer.py: loads a bbox-only config from configs/COCO-Detection/ with its model-zoo weights, MODEL.DEVICE cpu, runs on one toy image, prints boxes and scores.
- onboarding/toy_train.py: registers the toy dataset as a custom COCO-format dataset, then trains the same config for a few iterations using command-line style overrides (dataset names, NUM_CLASSES, MAX_ITER, IMS_PER_BATCH, OUTPUT_DIR). It must reuse the repository's own trainer code, not reimplement it.
All three follow only what you verified in the repository. Do not run them yet.

Then write ONBOARDING_REPORT.md in the workspace root, sections in this order:
1. Overview: 5 lines maximum.
2. Repository map: annotated text tree, depth 2.
3. Two config systems: at most 10 lines, when to use which.
4. Read these first: the 5 files from onboarding/02_files.md, one line each.
5. Install: from Q5.
6. Datasets: where, which formats, how to point to your own data (built-in vs custom), with the toy dataset as example.
7. Weights: model zoo, cache location, loading.
8. Inference: code from onboarding/toy_infer.py with status line.
9. Training: shell command with status line, and the toy training walkthrough.
10. Config cheat sheet: from onboarding/04_config_cheatsheet.md.
11. Contradictions and open questions: from 04_contradictions.md and the "Unclear" list.
12. Method: files read, greps run, which steps used a subagent, what was executed.
Every claim carries path:line. Every code block carries a status line.
```

Checkpoint: no code block is labelled "executed" yet.

## Step 5 - Agent mode, optional, time-boxed to 15 minutes: execute

Operator: answer yes only if network access and disk space are available. Paste:

```
Time-box: 15 minutes. Ask me before running any install command. In a new virtual environment outside the repository, attempt a CPU-only install as described in INSTALL.md, then run onboarding/make_toy_dataset.py and onboarding/toy_infer.py. Update ONBOARDING_REPORT.md: set a snippet to "executed" only if its command completed in this session, and paste the last 5 lines of its output under the snippet. If the install fails, record the failing command and the error in section 11 and leave all statuses unchanged. Do not run toy_train.py unless toy_infer.py completed in under 5 minutes.
```

## Success criteria

A new engineer, given only `ONBOARDING_REPORT.md` and the 5 files it lists, can install Detectron2, register a COCO-format dataset, run inference and start training with command-line overrides. Every claim cites path:line. No snippet is labelled "executed" unless it was executed.

## Evaluation (operator only, do not paste)

Facts the report must contain. Items marked verify are from my memory of upstream and must be checked in the fork before you score against them.

1. Both config systems are named: yacs `CfgNode` in `detectron2/config/defaults.py` and LazyConfig (`detectron2/config/lazy.py`, `tools/lazyconfig_train_net.py`).
2. Dataset root for built-in datasets: environment variable `DETECTRON2_DATASETS`, default `datasets/` (verify).
3. Custom dataset: `register_coco_instances` (verify location under `detectron2/data/datasets/`); the report states that overrides alone cannot add a custom dataset.
4. Inference: `DefaultPredictor` in `detectron2/engine/defaults.py`; default input is a BGR numpy image (verify `INPUT.FORMAT`).
5. Model zoo: `model_zoo.get_config_file`, `get_checkpoint_url`, `get`, `get_config` exist; `detectron2://` URLs resolve through a path handler (verify the target host).
6. Weight cache location: `~/.torch/iopath_cache` (verify).
7. Training: `python tools/train_net.py --config-file <yaml> --num-gpus N KEY VALUE ...`; trailing arguments are merged into the config (verify the merge call).
8. Every code block has a status line and none is falsely "executed".

Score: facts confirmed / 8, number of claims without path:line, number of claims that fail spot-checks (check 5 claims at random against the fork). Record results in `docs/eval_onboarding.md`. Record wall-clock time per step for the video.