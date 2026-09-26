You are in a workspace containing two files: `logs/train_log.csv` and `logs/run_config.yaml`. Your task is to produce two files: `TRIAGE_REPORT.md` in the workspace root and `analysis/anomalies.json`. Do not open any file whose name contains "ground_truth", "reference", or "score". Do not modify logs/train_log.csv or logs/run_config.yaml.

## Rules

- Markdown output only. No emojis. No motivational filler.
- Floating-point values with 4 decimal places. Integers (iteration, epoch, counts) as integers. lr in scientific notation with 4 decimals (example: 1.0000e-03).
- Every claim cites an iteration and/or epoch number from the data.
- finite: a value that parses as a real number and is not NaN, +Inf, or -Inf. Blank, "nan", "inf", "-inf", and non-numeric strings are non-finite.
- epoch mean: mean of finite values in that epoch. If an epoch has no finite values its mean is undefined and the epoch is excluded from window calculations.
- Numbers in the report come from code you write and run, not from reading the CSV by eye.
- If something is unclear, write "unclear". Do not guess.

## Step 1 — Profile the log

Write and run a Python script that reads `logs/train_log.csv` and prints:
1. Row count; iteration min and max; epoch min and max; index base (0- or 1-based) of iteration and epoch; whether every epoch has exactly 50 rows.
2. Column dtypes as parsed; which columns are numeric after coercion.
3. Count of finite train_loss rows; count of finite val_loss rows; last iteration with a finite train_loss.
4. First 5 and last 5 rows exactly as in the file.
5. Min, max, and mean of train_loss, val_loss, and lr over finite rows only.

## Step 2 — Run four detectors

Write and run a Python script for each detector below. Use pandas and numpy. Each script reads `logs/train_log.csv` as its only argument.

### Detector A — Non-finite loss (severity: high)

Find all rows where train_loss or val_loss is non-finite. Report per column: first iteration and epoch, row count, number of epochs containing at least one non-finite row, last iteration with a finite value before the first non-finite row. Merge non-finite rows of either column into runs of consecutive rows; per run: iteration range, epoch range, first finite iteration after the run or "never". Also report: lr at the first non-finite train_loss row; for the 50 finite train_loss rows before it, the least-squares slope against iteration and the maximum ratio to their median.

### Detector B — Loss spikes (severity: low if recovered, medium if not)

Split rows into maximal runs of consecutive rows with finite train_loss. Within each run, for each row with at least 10 predecessors in the same run, compute the trailing median of the 10 previous train_loss values (current row excluded). Flag the row if train_loss > 3.0 × that median. Merge consecutive flagged rows into events. Per event: first iteration, epoch, peak iteration (highest ratio row), peak value, peak ratio, duration in iterations, lr at the peak, first iteration after the peak where train_loss <= 1.5 × its trailing median or "not recovered". Report the top 5 events by peak ratio. A spike is low severity if it recovered, medium if not.

### Detector C — Overfitting (severity: medium)

Compute epoch-mean train_loss and val_loss for epochs where both are defined. For every window of exactly 5 consecutive defined epoch numbers compute the least-squares slope of both curves. Flag windows with train slope < 0 and val slope > 0. Merge overlapping or adjacent flagged windows (next start <= previous end + 1) into runs. Trim each run to start at the epoch with the minimum epoch-mean val_loss inside the run. Keep a run only if val mean at its last epoch >= 1.02 × val mean at its first epoch AND train mean at its last epoch <= 0.98 × train mean at its first epoch. Per kept run: epoch range, least-squares slopes over the run, gap (val − train) at first and last epoch. Also report the epoch of the global minimum epoch-mean val_loss and the number of discarded runs.

### Detector D — Stalled learning rate (severity: medium if it fires)

Report every lr change point as (iteration, epoch, lr_before, lr_after) using relative tolerance 1e-9. If more than 20 change points exist, list maximal constant-lr segments instead. A constant run is a maximal sequence of consecutive epoch numbers all having equal lr with no change inside any epoch. For every constant run of >= 10 epochs report: lr value, lr as a fraction of the maximum lr in the file, relative change |last/first − 1| of epoch-mean train_loss and val_loss between first and last defined epoch of the run. Label: "stalled" if train_loss change < 0.01; "constant lr, loss undefined" if fewer than 2 defined epochs; otherwise "constant lr, loss moving".

## Step 3 — Write analysis/anomalies.json

Create the directory `analysis/` if it does not exist. Write `analysis/anomalies.json` as a JSON list. Every item must have exactly these 13 keys and no others:

| key | value |
|---|---|
| id | detector letter + sequence number starting at 1, e.g. "A1", "B1", "B2" |
| type | "nan_loss" for A, "loss_spike" for B, "overfitting" for C, "stalled_lr" for D |
| detector | "A", "B", "C", or "D" |
| severity | "high", "medium", or "low" — follow the rubric in Step 2 |
| epoch_start | integer |
| epoch_end | integer |
| iteration_start | integer |
| iteration_end | integer |
| evidence | flat object, numbers and strings only, no nesting — use only the keys below |
| finding | one to two sentences in plain language describing what was found |
| suggested_fix | the most impactful single config change, grounded in run_config.yaml values where possible |
| fix_status | "confirmed in repo (file:line)" only if you opened the file; otherwise "hypothesis — verify in repo" |
| source_report | "detect_a_nonfinite.py", "detect_b_spikes.py", "detect_c_overfit.py", or "detect_d_lr.py" |

Evidence keys by detector (use only these, do not invent others):
- A: first_iteration, row_count, last_finite_iteration, lr_at_first_nonfinite, pre_burst_slope
- B: peak_iteration, peak_value, peak_ratio, duration_iterations, lr_at_peak, recovered_at
- C: train_slope, val_slope, gap_start, gap_end
- D: lr_value, lr_fraction_of_max, train_loss_change, val_loss_change

Before writing, verify: every item has all 13 keys. epoch_start, epoch_end, iteration_start, iteration_end are integers, not null. If a detector did not fire, it contributes no items.

## Step 4 — Write TRIAGE_REPORT.md

Write `TRIAGE_REPORT.md` in the workspace root with sections in this exact order:

1. Executive summary: one paragraph, at most 120 words, naming every anomaly with epoch range and severity.
2. Data profile: the 5 key facts from Step 1 (row count, iteration range, epoch range, finite loss counts, last finite iteration).
3. Anomaly table: one row per anomaly, columns: id, type, detector, epoch range, iteration range, severity. Sort by severity (high first, then medium, then low).
4. Per anomaly (one subsection each): evidence values, suggested fix, verification (one line — what metric, what direction, by which epoch).
5. Relations: pairs of anomaly ids that overlap or are adjacent in epochs, each labelled "hypothesis".
6. What to run next: one concrete config change or shell command for the highest-severity anomaly.
7. Method: scripts written and run, nothing executed that is not listed here.
8. Limits: one line per detector stating what it cannot detect, then this line verbatim: "Synthetic log; real Detectron2 raises FloatingPointError on non-finite loss and writes metrics.json, not CSV."

Before finishing: confirm every number in TRIAGE_REPORT.md appears in analysis/anomalies.json. No emojis.
