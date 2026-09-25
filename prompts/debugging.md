# ML-Dive: Debugging scenario for IBM Bob 2.0

## Scenario

A training run has finished. The metrics look wrong but the engineer does not know where. They have `logs/train_log.csv`, the run config `logs/run_config.yaml`, and a Detectron2 fork. They need a triage report: what went wrong, when, and what to change.

Log format: `iteration, epoch, train_loss, val_loss, lr, timestamp`. Expected: 5000 rows, 100 epochs, 50 iterations per epoch.

## Preconditions (operator, not for Bob)

- Workspace: the Detectron2 fork. Create `logs/` in it and copy exactly two files there: `train_log.csv` and `run_config.yaml`. Do not copy `generate_logs.py`, `reference_detectors.py`, `score_anomalies.py` or `docs/`. Do not open the ml-dive repo in that workspace.
- Before the run, execute `python logs/reference_detectors.py --json` in the ml-dive repo. It writes `logs/reference_anomalies.json`, the answer key for both the numbers and the final structural diff.
- Python 3 with pandas and numpy in Bob's terminal. If pandas is missing, Bob uses stdlib `csv` and `statistics`.
- Mode names below (Plan, Agent) follow the hackathon material. Verify they match the IDE.
- Outputs: intermediate files in `analysis/`, final `TRIAGE_REPORT.md` and `analysis/anomalies.json` in the workspace root / `analysis/`. After the run, copy `TRIAGE_REPORT.md`, `analysis/anomalies.json` and `analysis/anomalies.md` into `docs/examples/debugging/` in the ml-dive repo.

## Step 1 - Plan mode: rules and analysis plan

Mode: Plan. Two pastes.

Paste 1, session rules. If Bob supports a project rules file, put this block there instead.

```
Save the following rules verbatim to analysis/RULES.md, then follow them for the whole session.
- Output: Markdown. No emojis, no motivational filler.
- Floating-point values with 4 decimal places. Integers (iteration, epoch, counts) as integers. lr in scientific notation with 4 decimals (example: 1.0000e-03).
- Every claim cites an iteration and/or epoch, or file:line for code.
- Do not open logs/generate_logs.py or any file whose name contains "ground_truth" or "reference". Ignore them if they exist.
- finite: a value that parses as a real number and is not NaN, +Inf or -Inf. Blank, "nan", "inf", "-inf" and non-numeric strings are non-finite.
- epoch mean: mean over the finite values of that epoch. If an epoch has no finite value, its mean is undefined and the epoch is excluded from windows.
- Report the index base (0- or 1-based) of iteration and epoch as found in the file. Do not renumber.
- Every detector is a Python script in analysis/ and you record the command that runs it. Numbers in reports come from script output, not from reading the CSV by eye.
- Tag a claim "hypothesis — verify in repo" only if it is about code you did not open in this session. Claims from files you opened cite file:line and stand alone, without caveats.
- If something is unclear, write "unclear". Do not guess.
```

Paste 2, the plan.

```
Do not open logs/train_log.csv yet and do not run code. Write the analysis plan to analysis/PLAN.md with exactly four sections. Copy the thresholds below exactly; do not change any of them.

Section 1 - Segmentation. How rows map to epochs, how to verify exactly 50 rows per epoch, how to detect 0- vs 1-based indexing.

Section 2 - Detector specs. Severity for each detector is fixed: A is high, C is medium, D is medium if it fires. B is low if the spike recovers (train_loss returns to <= 1.5x trailing median), medium if it does not.
A. Non-finite. Rows where train_loss or val_loss is non-finite. Per column: first iteration and epoch, row count, number of epochs with at least one such row, last iteration with a finite value before the first such row. Merge non-finite rows of either column into runs of consecutive rows; per run: iteration range, epoch range, first finite iteration after the run or "never". Also: lr at the first non-finite train_loss row; for the 50 finite train_loss rows before it, the least-squares slope against iteration and the maximum ratio to their median.
B. Spike. Split the rows into maximal runs of consecutive rows with finite train_loss and process each run separately. In a run, for each row that has at least 10 predecessors in the same run, take the trailing median of the 10 previous train_loss values (current row excluded). Flag the row if train_loss > 3.0 x that median. Merge consecutive flagged rows into events. Per event: first iteration, epoch, peak iteration (the row with the highest ratio), peak value, peak ratio, duration in iterations, lr at the peak, and the first iteration after the peak in the same run where train_loss <= 1.5 x its trailing median, or "not recovered". Report the top 5 events by peak ratio.
C. Overfitting. Epoch-mean train_loss and val_loss over epochs where both are defined. For every window of exactly 5 consecutive epoch numbers, all defined, compute the least-squares slope of both curves. Flag windows with train slope < 0 and val slope > 0. Merge overlapping or adjacent flagged windows (next start <= previous end + 1) into runs. Trim each run to start at the epoch with the minimum epoch-mean val_loss inside the run; the run ends at its last epoch. Keep a run only if val mean at its last epoch >= 1.02 x val mean at its first epoch AND train mean at its last epoch <= 0.98 x train mean at its first epoch. Per kept run: epoch range, least-squares slopes of both curves over the run, gap (val - train) at the first and last epoch. Also report the epoch of the minimum epoch-mean val_loss over all defined epochs, the val mean at the last defined epoch, and the number of discarded runs.
D. LR. Report every lr change point as (iteration, epoch, lr before, lr after); equality uses relative tolerance 1e-9. If there are more than 20 change points, list the maximal constant-lr segments instead. An epoch is "constant" if all its rows have equal lr; epochs with a change inside are excluded. A constant run is a maximal run of consecutive epoch numbers that are constant and share the same lr. For every constant run of >= 10 epochs report: lr, lr as a fraction of the maximum lr in the file, and for train_loss and val_loss the relative change |last / first - 1| of the epoch mean between the first and last defined epoch of the run. Label the run "stalled" if the train_loss change is below 0.01; "constant lr, loss undefined" if the run has fewer than 2 defined epochs; otherwise "constant lr, loss moving".

Section 3 - Subagent contract. Script, output file and command for each detector: analysis/detect_a_nonfinite.py -> analysis/report_a.md, analysis/detect_b_spikes.py -> analysis/report_b.md, analysis/detect_c_overfit.py -> analysis/report_c.md, analysis/detect_d_lr.py -> analysis/report_d.md, each run as python <script> logs/train_log.csv. Every report ends with a line "Limits:" stating what that detector cannot see.

Section 4 - Edge cases. For each detector, where it can fail (non-finite rows, per-iteration lr, epochs with fewer than 50 rows, a constant-lr run whose loss is undefined) and how the spec above handles it.
```

Checkpoint: `analysis/RULES.md` and `analysis/PLAN.md` exist; PLAN.md has exactly 4 sections and contains the thresholds 3.0, 10, 1.5, 5, 1.02, 0.98, 10, 0.01 unchanged.

## Step 2 - Agent mode: profile and detect

Mode: Agent. Paste:

```
Read analysis/RULES.md and analysis/PLAN.md.

First write and run analysis/profile_log.py on logs/train_log.csv. Save its output to analysis/profile.md with:
1. Number of rows; iteration min and max; epoch min and max; index base of iteration and of epoch; whether every epoch has exactly 50 rows.
2. Column dtypes as parsed and which columns are numeric after coercion.
3. Number of finite train_loss rows, number of finite val_loss rows, last iteration with a finite train_loss.
4. The first 5 and last 5 rows, exactly as in the file.
5. Min, max and mean of train_loss, val_loss and lr over finite rows only.
If PLAN.md assumes something the data contradicts (row count, index base, rows per epoch), state which one and update PLAN.md before continuing.

Then spawn four subagents, one per detector in PLAN.md section 2 (A, B, C, D). Each subagent first reads analysis/RULES.md and analysis/PLAN.md, implements exactly its spec with the script name from section 3, runs it, writes its report to the file named in section 3, and returns only the path of that file. Subagents must not read each other's outputs and must not change thresholds.
```

Fallback if subagents are unavailable. Paste:

```
Subagents are not available. Implement and run the four detectors yourself, A then B then C then D, each with the script name and output file from analysis/PLAN.md section 3, without changing thresholds. Each report ends with a line "Limits:".
```

Checkpoint: `analysis/profile.md` states the index base and the last finite train_loss iteration; `analysis/report_a.md` to `analysis/report_d.md` exist.

## Step 3 - Agent mode: consolidate into anomalies.json

Mode: Agent. Paste:

```
Read docs/anomalies_schema.md. Read analysis/report_a.md to analysis/report_d.md.

Write analysis/anomalies.json: a JSON list, one item per detected anomaly, matching docs/anomalies_schema.md exactly. Use only the evidence keys listed there for each detector; do not invent others. id is the detector letter plus a sequence number starting at 1 (A1, B1, B2, ...). type is loss_spike for B, overfitting for C, nan_loss for A, stalled_lr for D. severity follows the rubric in PLAN.md section 2. source_report is the path to the report file this anomaly came from. Leave finding, suggested_fix and fix_status as placeholders for now: finding is one sentence from the evidence, suggested_fix is "TBD", fix_status is "TBD".

Also write analysis/anomalies.md: one table sorted by severity (high, medium, low), ties by first epoch. Columns: id, type, detector, epoch range, iteration range, evidence (rendered as key=value, comma separated), severity. Then a section "Relations": pairs of anomaly ids that overlap or are adjacent in epochs, each labelled "hypothesis" with the numbers that motivate it. If two reports contradict each other, say so; do not resolve it silently.

Check before finishing: every item in anomalies.json has all 13 keys from docs/anomalies_schema.md; anomalies.md and anomalies.json list the same anomalies in the same order.
```

Checkpoint: `analysis/anomalies.json` validates against the 13 keys; `analysis/anomalies.md` table matches it row for row.

## Step 4 - Agent mode: fixes grounded in the repository

Mode: Agent. Paste:

```
Read logs/run_config.yaml and analysis/anomalies.json. For each item, search the repository (grep) and open the file before naming any symbol or config key. Candidate locations to check, not to assume: detectron2/config/defaults.py (SOLVER.BASE_LR, SOLVER.WARMUP_ITERS, SOLVER.WARMUP_FACTOR, SOLVER.STEPS, SOLVER.GAMMA, SOLVER.LR_SCHEDULER_NAME, SOLVER.CLIP_GRADIENTS.*, SOLVER.AMP.ENABLED, SOLVER.WEIGHT_DECAY, SOLVER.IMS_PER_BATCH, TEST.EVAL_PERIOD), detectron2/solver/build.py, detectron2/solver/lr_scheduler.py, detectron2/engine/train_loop.py, detectron2/engine/hooks.py, detectron2/engine/defaults.py, detectron2/data/transforms/. If a candidate does not exist, say so.

For each item in analysis/anomalies.json, fill in:
- finding: rewrite as one to two full sentences using the evidence values, referencing file:line where the finding implicates specific code (for example, which config key governs the behavior).
- suggested_fix: file and config key or code location with file:line, current value from logs/run_config.yaml, proposed value, and why. At most two changes per anomaly, most impactful first.
- fix_status: "confirmed in repo (file:line)" only if every file:line in suggested_fix was opened this session, otherwise "hypothesis — verify in repo".
Write the updated list back to analysis/anomalies.json, same order, same ids, all 13 keys still present. Also update analysis/anomalies.md's evidence column if any numbers changed; do not change severities.

Also answer, with file:line: what does the trainer do when the loss becomes non-finite?
```

Checkpoint: no item has `finding`, `suggested_fix` or `fix_status` equal to "TBD"; every "confirmed in repo" status corresponds to a file opened in this session.

## Step 5 - Agent mode: triage report

Mode: Agent. Paste:

```
Write TRIAGE_REPORT.md in the workspace root from analysis/anomalies.json and analysis/profile.md. Sections in this order:
1. Executive summary: one paragraph, at most 120 words, naming every anomaly with epoch range and severity.
2. Data profile: 5 lines from profile.md.
3. Anomaly table: identical to analysis/anomalies.md, same order.
4. Per anomaly: Evidence (from the evidence object), Suggested fix, Verification (one line: what metric, what direction, by which epoch).
5. Relations between anomalies: from analysis/anomalies.md's Relations section, hypotheses only.
6. What to run next: one shell command or one config change for the most severe anomaly. If it is a training command, use Detectron2 override syntax (python tools/train_net.py --config-file <file> KEY VALUE) and name the config file. If the real training config is not in the repo, tag the command "hypothesis — verify in repo".
7. Method: scripts in analysis/ with the command that ran each; which steps used a subagent; anything not executed.
8. Limits: the four "Limits:" lines from the detector reports, then this line verbatim: Synthetic log; real Detectron2 raises FloatingPointError on non-finite loss and writes metrics.json, not CSV. The harness validates detectors against known ground truth.
Before finishing, check: every number in the report appears in analysis/anomalies.json; no emoji.
```

## Success criteria

An ML engineer who has never seen this log can, after reading `TRIAGE_REPORT.md`: (a) name every anomaly, (b) point to the exact epoch and iteration, (c) make one concrete change to the training config for the most severe issue, and (d) tell which claims rest on opened code and which are hypotheses. `analysis/anomalies.json` validates against `docs/anomalies_schema.md`.

## Evaluation (operator only, do not paste)

Run:
```
python logs/reference_detectors.py --json
python logs/score_anomalies.py analysis/anomalies.json logs/reference_anomalies.json
```
`score_anomalies.py` prints recall, false positives and severity agreement, comparing Bob's `analysis/anomalies.json` against the reference. Record its output in `docs/eval_debugging.md`, along with wall-clock time per step. Do not edit the prompts to fit one run's output without re-running Steps 1-5 in a fresh session.