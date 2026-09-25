"""Verify findings #4-#7.

  #4 redundant `import numpy as np` inside build_panel  -> gone, charts still render
  #5 unused ANOMALIES_JSON_NAME / TRAIN_LOG_CSV_NAME    -> gone
  #6 spike-zoom TypeError when a loss_spike has no epoch range -> guarded
  #7 `epoch_end` falsy-coalesce dropped a legitimate 0   -> explicit None checks
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "venv" / "Lib" / "site-packages"))
sys.path.insert(0, str(ROOT))

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))


import ml_dive as m  # noqa: E402

# --------------------------------------------------------------- #4 and #5 --
_src = Path(m.__file__).read_text(encoding="utf-8")
check("#4 exactly one `import numpy as np` (module level)",
      _src.count("import numpy as np") == 1,
      f"count={_src.count('import numpy as np')}")
check("#4 module-level `np` present", hasattr(m, "np") and m.np is not None)
check("#5 ANOMALIES_JSON_NAME removed", not hasattr(m, "ANOMALIES_JSON_NAME"))
check("#5 TRAIN_LOG_CSV_NAME removed", not hasattr(m, "TRAIN_LOG_CSV_NAME"))
check("#5 report-name constants kept",
      hasattr(m, "ONBOARDING_REPORT_NAME") and hasattr(m, "TRIAGE_REPORT_NAME"))

# ------------------------------------------------------------------------ #7 --
def zone_payload(**over):
    base = {
        "id": "Z0", "type": "loss_spike", "detector": "T",
        "severity": "high",
        "epoch_start": 1, "epoch_end": 3,
        "iteration_start": 1025, "iteration_end": 1026,
        "evidence": {"peak_ratio": 7.882},
        "finding": "x", "suggested_fix": "y",
        "fix_status": "hypothesis", "source_report": "TRIAGE_REPORT.md",
    }
    base.update(over)
    return json.dumps([base]).encode()


raw = zone_payload(epoch_start=0, epoch_end=0)
zs = m.load_diagnosis("t.json", len(raw), raw)
check("#7 parsed without dropping", zs is not None and len(zs) == 1, str(zs))
if zs:
    z = zs[0]
    check("#7 epoch_start 0 preserved", z.get("epoch_start") == 0, repr(z.get("epoch_start")))
    check("#7 epoch_end 0 preserved", z.get("epoch_end") == 0, repr(z.get("epoch_end")))

# legacy end_epoch=0 must also survive
raw = zone_payload(epoch_end=None, end_epoch=0)
zs = m.load_diagnosis("t.json", len(raw), raw)
check("#7 legacy end_epoch 0 preserved",
      zs and zs[0].get("epoch_end") == 0, repr(zs[0].get("epoch_end") if zs else None))

# a genuinely absent epoch_end still falls back to epoch_start
raw = zone_payload(epoch_end=None)
zs = m.load_diagnosis("t.json", len(raw), raw)
check("#7 absent epoch_end falls back to epoch_start",
      zs and zs[0].get("epoch_end") == 1, repr(zs[0].get("epoch_end") if zs else None))

# ------------------------------------------------------------------------ #6 --
# a loss_spike carrying ONLY an iteration range (schema allows this)
raw = zone_payload(epoch_start=None, epoch_end=None)
zs_noep = m.load_diagnosis("t.json", len(raw), raw)
check("#6 spike zone with only an iteration range is kept",
      zs_noep is not None and len(zs_noep) == 1, str(zs_noep))

# ------------------------------------------------------------------ end-to-end --
from streamlit.testing.v1 import AppTest  # noqa: E402

CSV = (ROOT / "logs" / "train_log.csv").read_bytes()
LOG = (ROOT / "logs" / "detectron2_run.log").read_bytes()
REF = (ROOT / "logs" / "reference_anomalies.json").read_bytes()


def find(widgets, key):
    for w in widgets:
        if getattr(w, "key", None) == key:
            return w
    return None


at = AppTest.from_file(str(ROOT / "ml_dive.py"), default_timeout=60)
at.run()
check("app runs at startup", not at.exception, str(at.exception))

mode = find(at.selectbox, "mode_select") or at.selectbox[0]
mode.set_value("ML Training Log Debugger")
at.run()
intake = find(at.file_uploader, "ml_dive_uploader") or at.file_uploader[0]
intake.set_value([("train_log.csv", CSV, "text/csv"),
                  ("detectron2_run.log", LOG, "text/plain")])
at.run()
check("intake upload clean", not at.exception, str(at.exception))


def all_text():
    parts = [str(getattr(w, "value", "")) for w in at.markdown]
    parts += [str(getattr(w, "label", "")) for w in getattr(at, "expander", [])]
    return "\n".join(parts)


# --- A: reference (epochs present) must still show the spike zoom ------------
diag = find(at.file_uploader, "dbg_diag")
diag.set_value(("reference_anomalies.json", REF, "application/json"))
at.run()
check("reference diagnosis clean", not at.exception, str(at.exception))
check("#6 normal spike zoom still renders (no regression)",
      "Zoom" in all_text(), f"missing; text head={all_text()[:400]!r}")

# --- B: loss_spike with no epoch range must NOT crash ------------------------
payload = json.loads(REF.decode("utf-8"))
items = payload["anomalies"] if isinstance(payload, dict) else payload
mod = deepcopy(items)
for it in mod:
    if "spike" in str(it.get("type", "")).lower():
        it["epoch_start"] = None
        it["epoch_end"] = None
mod_raw = json.dumps(mod).encode()

# widget references are recreated on every run — re-fetch before re-setting
diag = find(at.file_uploader, "dbg_diag")
diag.set_value(("no_epoch_spike.json", mod_raw, "application/json"))
at.run()
print("\n  --- diagnosis captions after 2nd upload ---")
for ln in all_text().splitlines():
    if "Diagnosis" in ln or "no_epoch" in ln or "Zoom" in ln:
        print("   ", ln[:220])
print(f"  --- uploaders: {[getattr(u,'key',None) for u in at.file_uploader]} ---")
print("  --- end ---\n")
check("#6 epoch-less spike zone does not crash", not at.exception, str(at.exception))
_zoom_lines = [ln for ln in all_text().splitlines() if "Zoom" in ln]
print("\n  --- lines containing 'Zoom' ---")
for ln in _zoom_lines:
    print("   ", ln[:220])
print("  --- end ---\n")
check("#6 epoch-less spike hides the epoch zoom", not _zoom_lines,
      f"found {len(_zoom_lines)} line(s)")

# charts still draw for the epoch-less zone
check("#6 charts still render",
      len([f for f in at.dataframe if hasattr(f.value, "columns")]) >= 1)

# --- C: all-NaN epoch column must not divide by zero -------------------------
# (the peak_iteration fallback).  Covered by the guard itself; assert via source.
src = Path(m.__file__).read_text(encoding="utf-8")
check("#7 peak_estimate guards nunique() == 0",
      'if n_epochs else 1' in src, "guard missing")
check("#7 peak_estimate clamped to >= 1", "max(1, int((z[" in src, "clamp missing")

print("=" * 68)
fails = 0
for name, ok, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"  -> {detail}"))
    fails += (not ok)
print("=" * 68)
print(f"  {len(results) - fails}/{len(results)} passed")
sys.exit(1 if fails else 0)
