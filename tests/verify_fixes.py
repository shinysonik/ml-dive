"""Verify the two fixes without a browser, using Streamlit's AppTest."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "venv" / "Lib" / "site-packages"))

from streamlit.testing.v1 import AppTest

CSV = (ROOT / "logs" / "train_log.csv").read_bytes()
LOG = (ROOT / "logs" / "detectron2_run.log").read_bytes()
JSON = (ROOT / "logs" / "reference_anomalies.json").read_bytes()

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))


def find(widgets, key):
    """Locate a widget by its session_state key, falling back to index order."""
    for w in widgets:
        if getattr(w, "key", None) == key:
            return w
    return None


at = AppTest.from_file(str(ROOT / "ml_dive.py"), default_timeout=60)
at.run()
check("app runs clean at startup", not at.exception, str(at.exception))

# --- switch to the debugger mode -------------------------------------------
mode = find(at.selectbox, "mode_select") or at.selectbox[0]
mode.set_value("ML Training Log Debugger")
at.run()
check("mode switch clean", not at.exception, str(at.exception))

# --- upload the CSV + text log ----------------------------------------------
intake = find(at.file_uploader, "ml_dive_uploader") or at.file_uploader[0]
intake.set_value(
    [
        ("train_log.csv", CSV, "text/csv"),
        ("detectron2_run.log", LOG, "text/plain"),
    ]
)
at.run()
check("intake upload clean", not at.exception, str(at.exception))

# --- attach the diagnosis JSON ----------------------------------------------
diag = find(at.file_uploader, "dbg_diag")
check("dbg_diag uploader found", diag is not None, f"uploaders={[getattr(u,'key',None) for u in at.file_uploader]}")
if diag is not None:
    diag.set_value(("reference_anomalies.json", JSON, "application/json"))
    at.run()
    check("diagnosis upload clean", not at.exception, str(at.exception))

    md = [m.value for m in at.markdown]
    alltext = "\n".join(str(v) for v in md)

    # FIX #2 assertions — these were permanently dead before the change
    check(
        "logs-tab cross-reference renders",
        "Cross-reference: log signals" in alltext,
        "missing 'Cross-reference: log signals'",
    )
    check(
        "'Correlated Anomaly' column present",
        any(
            "Correlated Anomaly" in list(f.value.columns)
            for f in at.dataframe
            if hasattr(f.value, "columns")
        ),
        f"tables={[list(f.value.columns) for f in at.dataframe if hasattr(f.value, 'columns')]}",
    )
    check(
        "metric-tables cross-ref banner renders",
        "Log ↔ Metric Cross-Reference" in alltext,
        "missing banner",
    )
    check(
        "diagnosis cards show a log footprint",
        "Log footprint" in alltext,
        "missing log footprint",
    )

    # FIX #1 / audit regression checks
    # Diagnostics: dump every rendered table's columns
    print("\n  --- rendered tables ---")
    for i, f in enumerate(at.dataframe):
        try:
            v = f.value
            print(f"    [{i}] columns={list(v.columns)!r} rows={len(v)}")
        except Exception as e:
            print(f"    [{i}] <unreadable: {e!r}>")
    print("  --- end tables ---\n")

    check(
        "numpy import OK (audit renders)",
        any("non-finite value(s)" in str(getattr(w, "value", "")) for w in at.warning),
        "non-finite audit warning absent",
    )
    check(
        "audit reports 300",
        any("300 non-finite" in str(getattr(w, "value", "")) for w in at.warning),
        f"got {[str(getattr(w,'value','')) for w in at.warning]}",
    )
    check("no exception after all uploads", not at.exception, str(at.exception))

# --- report -----------------------------------------------------------------
print("=" * 68)
fails = 0
for name, ok, detail in results:
    mark = "PASS" if ok else "FAIL"
    fails += (not ok)
    print(f"  [{mark}] {name}" + ("" if ok else f"  -> {detail}"))
print("=" * 68)
print(f"  {len(results) - fails}/{len(results)} passed")
sys.exit(1 if fails else 0)
