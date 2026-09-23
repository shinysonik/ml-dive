# anomalies.json schema

One JSON list. Every item has these keys. No additional keys unless listed under "evidence keys by detector" below.

| key | type | notes |
|---|---|---|
| id | string | stable within one run, e.g. "A1". Detector letter + sequence number if a detector fires more than once. |
| type | string, one of loss_spike \| overfitting \| nan_loss \| stalled_lr | semantic category |
| detector | string, one of A \| B \| C \| D | which spec in prompts/debugging.md PLAN.md section 2 produced this |
| severity | string, one of high \| medium \| low | never "critical" — see rubric in prompts/debugging.md |
| epoch_start | integer | |
| epoch_end | integer | |
| iteration_start | integer | |
| iteration_end | integer | |
| evidence | object | flat, numbers and strings only, no nesting. Keys below. |
| finding | string | one to two sentences, plain language |
| suggested_fix | string | the change, grounded or not |
| fix_status | string, one of "confirmed in repo (file:line)" \| "hypothesis — verify in repo" | mandatory, never omitted |
| source_report | string | path to the analysis/report_*.md this came from, or "reference_detectors.py" for the answer key |

## evidence keys by detector

Detector A (nan_loss): `first_iteration`, `row_count`, `last_finite_iteration`, `lr_at_first_nonfinite`, `pre_burst_slope`.
Detector B (loss_spike): `peak_iteration`, `peak_value`, `peak_ratio`, `duration_iterations`, `lr_at_peak`, `recovered_at`.
Detector C (overfitting): `train_slope`, `val_slope`, `gap_start`, `gap_end`.
Detector D (stalled_lr): `lr_value`, `lr_fraction_of_max`, `train_loss_change`, `val_loss_change`.

Do not add keys outside this list without updating this file first. The diff script in `logs/score_anomalies.py` assumes these names.

## Severity rubric

- high: training diverged or results are invalid (non-finite loss).
- medium: run is usable but final quality is degraded (overfitting; stalled lr with a loss plateau).
- low: transient and recovered (a spike that returns to <= 1.5x trailing median). A spike that does not recover is medium.

## Files that produce or consume this schema

- `analysis/anomalies.json` — written by Bob in prompts/debugging.md Step 3. This is the production output.
- `logs/reference_anomalies.json` — written by `logs/reference_detectors.py --json`. This is the answer key. `finding` is a short auto-generated sentence, `suggested_fix` is always `null`, `fix_status` is always `"not applicable — reference detector does not propose fixes"`.
- `logs/score_anomalies.py` — diffs any two files in this schema and prints recall / false positives / severity agreement.