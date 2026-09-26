# Bug Fix Plan — ml_dive.py (pre-Bob integration)

## Overview

Two bugs must be fixed before the demo. Both are crash-level issues visible to a
judge in the browser. A third issue (dead code) is a clean-up with zero risk. No
other changes are in scope.

Fixes are ordered by risk and dependency: fix the crash that can happen silently
first, then the one that requires a specific upload scenario, then the dead code.

---

## Sub-Task 1 — Guard `_zone_to_x` against `ee = None`

**Status:** [ ] pending

### Intent
`_zone_to_x` calls `float(ee)` at line 399 without first checking that `ee` is not
None. If a zone reaches this branch with `epoch_end=None`, the function raises
`TypeError`, which bubbles up through `render_anomaly_chart` and crashes the entire
chart panel with a Streamlit red error screen — the most visible part of the demo.

Although `load_diagnosis` normalises every zone it produces so that `epoch_end` is
never None when `epoch_start` is non-None (it sets `epoch_end = epoch_start` as
fallback at line 318–319), `_zone_to_x` is a standalone utility that takes a raw
dict. Any caller that constructs a zone directly (tests, future code, the e2e test
in verify_minor.py that explicitly passes `epoch_end=None`) can trigger this path.

The fix is a one-line extension of the existing `se` guard to also cover `ee`.

### Expected Outcomes
- `_zone_to_x` returns `None` (instead of crashing) for any zone where `epoch_end`
  is absent.
- verify_minor.py test `#6 epoch-less spike zone does not crash` continues to pass.
- verify_minor.py test `#6 epoch-less spike hides the epoch zoom` continues to pass.
- No other test changes.

### Todo List
1. In `_zone_to_x` at line 395–396, change the guard from:
   ```
   if se is None:
       return None
   ```
   to:
   ```
   if se is None or ee is None:
       return None
   ```
2. Confirm no other line in `_zone_to_x` calls `float(ee)` before the guard.

### Relevant Context
- `_zone_to_x`: ml_dive.py line 376–407
- Only call site: ml_dive.py line 483 (inside `build_panel`, result is already
  guarded with `if r is not None`)
- Related tests: tests/verify_minor.py lines 74–155 (the #6 block)

---

## Sub-Task 2 — Prevent `StopIteration` crash in `pick_by_name`

**Status:** [ ] pending

### Intent
`pick_by_name` uses a bare `next(generator)` with no default. If the name passed
by a `st.selectbox` widget does not match any file in the list (which can happen
on a Streamlit rerun during a file re-upload or mode switch), Python raises
`StopIteration`. Streamlit catches this as an unhandled exception and shows a
full red crash screen.

The fix has two parts:
1. Add a `None` default to `next()` in `pick_by_name`.
2. Add a `None` guard at each of the three call sites before the return value is
   dereferenced (`.getvalue()`, `.size`). Each call site already sits inside a
   conditional block, so an early return is the cleanest option.

### Expected Outcomes
- A selectbox/list desync on rerun produces no exception and no red screen.
- The three call sites handle a `None` return gracefully (early return or skip).
- All existing tests continue to pass — no test directly asserts `StopIteration`
  behaviour, and the e2e AppTest sessions always upload matching names so they
  never hit the None path.

### Todo List
1. Change `pick_by_name` at line 717 to:
   ```python
   return next((f for f in files if f.name == name), None)
   ```
2. At the call site in `render_onboarding` (line 949), add a guard immediately
   after the `pick_by_name` call:
   ```python
   if chosen is None:
       return
   ```
   (or `st.warning` + `return`, but a silent return is sufficient — the selectbox
   only shows names from the list, so this state is transient and resolves itself
   on the next rerun)
3. At the call site in `render_logs_panel` (line 997), add the same guard.
4. At the call site in `render_tables_panel` (line 1164), add the same guard.

### Relevant Context
- `pick_by_name`: ml_dive.py line 715–717
- Call sites: ml_dive.py lines 949, 997, 1164
- Surrounding context at each call site: the return value is used immediately for
  `.getvalue()` and `.size` — both would raise `AttributeError` on None without
  the guard
- No test directly tests `pick_by_name` in isolation; the e2e tests in
  verify_fixes.py and verify_minor.py always upload files whose names match the
  selectbox, so they are unaffected

---

## Sub-Task 3 — Remove dead `chart_df` construction when zones are present

**Status:** [ ] pending

### Intent
In `render_tables_panel`, lines 1198–1201 build a `chart_df` variable and
conditionally join an index column into it. When a diagnosis is loaded (`zones`
is not None), `render_anomaly_chart` is called instead of `st.line_chart`, and
`chart_df` is never used. The dead code causes no crash but adds confusion — a
reader might think `chart_df` is passed somewhere it is not.

The fix is to move the `chart_df` construction inside the `else` branch where it
is actually consumed.

### Expected Outcomes
- `chart_df` is only constructed in the branch where it is used (`st.line_chart`).
- The `render_anomaly_chart` branch is unchanged.
- All tests continue to pass — no test reads `chart_df` directly.

### Todo List
1. Remove the `chart_df` assignment and `if index_col: chart_df = chart_df.join(...)`
   lines from the shared block (lines 1198–1201).
2. Move those same two lines inside the `else` branch (the branch that calls
   `st.line_chart`), immediately before the `st.line_chart` call.

### Relevant Context
- `render_tables_panel`: ml_dive.py lines 1090–1229
- The `if zones` / `else` split: ml_dive.py lines 1202–1208
- `chart_df` is referenced only in the `else` branch at line 1208

---

## Risk Assessment

| Sub-Task | Crash risk of the fix itself | Tests that exercise the changed code |
|---|---|---|
| 1 — `_zone_to_x` guard | None — adds an early return, can never make existing passing paths crash | verify_minor.py #6 block |
| 2 — `pick_by_name` default | None — existing e2e sessions never hit the None path | verify_fixes.py, verify_minor.py, verify_c1b4b3b7.py e2e sessions |
| 3 — dead `chart_df` | None — moves code inside the branch where it was already used | verify_fixes.py (chart render checks) |
