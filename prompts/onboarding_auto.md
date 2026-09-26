You are in a workspace containing a cloned Git repository. Your task is to produce a single file: `ONBOARDING_REPORT.md` in the workspace root. Do not install anything. Do not run training or inference commands.

## Rules

- Markdown only. No emojis. No motivational filler.
- Every factual claim about the repository cites path:line, taken from a file you opened or a grep result in this session. Do not cite from memory.
- Separate "verified in code" from "stated in documentation". If they differ, report the difference.
- Tag a claim "hypothesis — verify in repo" only if it is about code you did not open in this session.
- If something is unclear, write "unclear". Do not guess.
- Every code block carries a status line: "executed", "static check only", or "hypothesis — verify in repo".

## What to do

Work through these steps in order. Do not skip any. Write all intermediate results directly into ONBOARDING_REPORT.md as you go — do not create separate files.

### Step 1 — Repository map

Read the top-level directory listing. Then read the main package directory one level deep. Write a text tree (depth 2) with one-line descriptions inferred from names.

List which subdirectory or file is responsible for each of: config system, model zoo, data loading and dataset registration, training loop, inference, checkpoint loading. If more than one config system exists, list all paths without concluding yet.

### Step 2 — Key files

Check whether each of these files exists. If a file is missing, say so and do not substitute:
README.md, setup.py, any INSTALL or GETTING_STARTED document, the primary config defaults file, the model zoo module, the data catalog, the dataset registration file, the data build file, the engine defaults file, the checkpoint loader, the main training script, the demo or inference script.

Open and read each file that exists. For each, record:
- Purpose: one sentence.
- Key symbols: up to 5 functions or classes with path:line.

After reading all files, list the 5 files a new engineer should read first, one line each with the reason.

### Step 3 — Answer the onboarding questions

Answer each question using only what you verified in Step 2. Cite path:line for every statement.

Q1 Data. Which config keys select datasets. How a dataset name maps to a loader. Which formats are built in. Where the dataset root comes from (environment variable or default). Minimum code to register a custom COCO-format dataset. Whether datasets can be relocated without editing code — answer separately for built-in and custom.

Q2 Weights. How the model zoo getter and checkpoint-URL helper work. How MODEL.WEIGHTS is resolved to a local file. Where downloaded weights are cached.

Q3 Inference. A minimal end-to-end example: config + weights + one image → predictions. Which class runs inference, expected image format, structure of the output dict.

Q4 Training. The minimal command-line invocation with a config file. How command-line overrides are parsed. What extra code is needed to train on a custom dataset beyond config overrides.

Q5 Install. Steps from the install document and setup.py. Hard requirements. The two most likely install failures.

End Q5 with a section titled "Unclear" listing everything you could not confirm from opened files.

## Output format

Write ONBOARDING_REPORT.md with sections in this exact order:

1. Overview (5 lines maximum)
2. Repository map (annotated text tree, depth 2)
3. Config systems (at most 10 lines — how many exist, when to use each)
4. Read these first (the 5 files from Step 2, one line each)
5. Install (from Q5)
6. Datasets (Q1 answer — built-in vs custom, dataset root, registration example)
7. Weights (Q2 answer — model zoo, cache location, loading)
8. Inference (Q3 answer — code block with status line)
9. Training (Q4 answer — command with status line)
10. Unclear (the list from Q5)
11. Method (files opened, greps run, nothing executed)

Every claim in sections 6–9 cites path:line. Every code block carries a status line. No code block is labelled "executed".
