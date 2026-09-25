"""Position-aware log <-> anomaly cross-reference verification.

Two layers:
  A. unit tests on _line_iteration / _iter_in_zone / _zones_for_line
  B. end-to-end AppTest asserting that each log line is credited to the
     RIGHT zone, not to every zone that wants its severity label.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "venv" / "Lib" / "site-packages"))
sys.path.insert(0, str(ROOT))

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))


# ---------------------------------------------------------------- A. units --
import ml_dive as m  # noqa: E402

IT = m._line_iteration
check("iter: 3001/5000            -> 3001", IT("iter: 3001/5000 total_loss: inf lr: 1.25e-04") == 3001)
check("'at iteration 3001.'       -> 3001", IT("ERROR: Metric NaN detected for 'loss' at iteration 3001.") == 3001)
check("'at iter 1025'            -> 1025", IT("WARNING: train_loss jumped 7.8820x at iter 1025 and recovered") == 1025)
check("'since iter 3951'          -> 3951", IT("WARNING: lr unchanged since iter 3951") == 3951)
check("'iterations 3001-3150'     -> 3001", IT("window closed (iterations 3001-3150, 150 rows)") == 3001)
check("'max_iter: 5000'           -> None", IT("max_iter: 5000 steps: [951, 1951]") is None)
check("'5000 iterations'          -> None", IT("Training took 2:46:32 (5000 iterations)") is None)
check("bare lr / timestamps       -> None", IT("d2.utils.events INFO: lr: 6.2500e-05") is None)

Z = {"iteration_start": 3001, "iteration_end": 3150}
check("3001 inside 3001-3150      -> True", m._iter_in_zone(Z, 3001))
check("3150 inside 3001-3150      -> True", m._iter_in_zone(Z, 3150))
check("3000 below range           -> False", not m._iter_in_zone(Z, 3000))
check("3151 above range           -> False", not m._iter_in_zone(Z, 3151))
check("None iteration             -> False", not m._iter_in_zone(Z, None))
check("NaN iteration              -> False", not m._iter_in_zone(Z, float("nan")))
check("zone without iter range    -> False", not m._iter_in_zone({"type": "x"}, 3001))

zones = [
    {"type": "loss_spike", "iteration_start": 1025, "iteration_end": 1026},
    {"type": "stalled_lr", "iteration_start": 3951, "iteration_end": 5000},
    {"type": "nan_loss", "iteration_start": 3001, "iteration_end": 3150},
    {"type": "overfitting", "iteration_start": 1951, "iteration_end": 3000},
]
t = lambda z: [x["type"] for x in z]
check("Warning @1025 -> only loss_spike", t(m._zones_for_line(1025, "Warning", zones)) == ["loss_spike"])
check("Warning @3951 -> only stalled_lr", t(m._zones_for_line(3951, "Warning", zones)) == ["stalled_lr"])
check("NaN       @3001 -> only nan_loss", t(m._zones_for_line(3001, "NaN metric", zones)) == ["nan_loss"])
check("Warning, no iter -> nothing", m._zones_for_line(None, "Warning", zones) == [])
check("unrelated severity -> nothing", m._zones_for_line(3001, "Traceback", zones) == [])

# ------------------------------------------------------------------ B. e2e --
from streamlit.testing.v1 import AppTest  # noqa: E402

CSV = (ROOT / "logs" / "train_log.csv").read_bytes()
LOG = (ROOT / "logs" / "detectron2_run.log").read_bytes()
JSON = (ROOT / "logs" / "reference_anomalies.json").read_bytes()


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
check("mode switch clean", not at.exception, str(at.exception))

intake = find(at.file_uploader, "ml_dive_uploader") or at.file_uploader[0]
intake.set_value(
    [("train_log.csv", CSV, "text/csv"), ("detectron2_run.log", LOG, "text/plain")]
)
at.run()
check("intake upload clean", not at.exception, str(at.exception))

diag = find(at.file_uploader, "dbg_diag")
check("dbg_diag uploader found", diag is not None)
diag.set_value(("reference_anomalies.json", JSON, "application/json"))
at.run()
check("diagnosis upload clean", not at.exception, str(at.exception))

alltext = "\n".join(str(m_.value) for m_ in at.markdown)

# locate the findings table
findings_df = None
for f in at.dataframe:
    try:
        cols = list(f.value.columns)
    except Exception:
        continue
    if "Correlated Anomaly" in cols:
        findings_df = f.value
        break

check("findings table found", findings_df is not None)
if findings_df is not None:
    check(
        "'iteration' column present",
        "iteration" in findings_df.columns,
        f"cols={list(findings_df.columns)}",
    )
    print("\n  --- findings table ---")
    print(findings_df.to_string(index=False))
    print("  --- end findings ---\n")

    def correlated(iteration, severity=None):
        d = findings_df
        if severity:
            d = d[d["severity"] == severity]
        rows = d[d["iteration"] == iteration]
        if rows.empty:
            return None
        return str(rows.iloc[0]["Correlated Anomaly"])

    spike = correlated(1025)
    check("spike line credited to loss_spike",
          spike is not None and "loss_spike" in spike, f"got {spike!r}")
    check("spike line NOT credited to stalled_lr",
          spike is not None and "stalled_lr" not in spike, f"got {spike!r}")

    stall = correlated(3951)
    check("stall line credited to stalled_lr",
          stall is not None and "stalled_lr" in stall, f"got {stall!r}")
    check("stall line NOT credited to loss_spike",
          stall is not None and "loss_spike" not in stall, f"got {stall!r}")

    nan_rows = findings_df[findings_df["severity"] == "NaN metric"]
    ok_nan = len(nan_rows) > 0 and all("nan_loss" in str(v) for v in nan_rows["Correlated Anomaly"])
    check("all NaN lines credited to nan_loss", ok_nan,
          f"vals={[str(v) for v in nan_rows['Correlated Anomaly']]}")

    # The batch-size retry warning names no iteration -> must claim nothing.
    retry = findings_df[findings_df["message"].str.contains("reduced batch size", na=False)]
    if len(retry):
        v = str(retry.iloc[0]["Correlated Anomaly"])
        check("unpositioned retry line claims nothing", v == "—", f"got {v!r}")
    else:
        check("unpositioned retry line claims nothing", True, "row not found (n/a)")

# cross-reference block must show separate per-zone counts
check("cross-ref renders", "Cross-reference: log signals" in alltext, "absent")
check(
    "cross-ref shows loss_spike with an in-range count",
    "loss_spike" in alltext and "in range" in alltext,
    "missing 'in range' wording",
)
check(
    "banner renders", "Log ↔ Metric Cross-Reference" in alltext, "absent"
)
check("log footprint renders", "Log footprint" in alltext, "absent")

print("\n  --- cross-ref lines ---")
for line in alltext.splitlines():
    if "in range" in line or "Log footprint" in line or "inside iterations" in line:
        print("   ", line[:200])
print("  --- end ---\n")

check("no exception after uploads", not at.exception, str(at.exception))

# ------------------------------------------------------------------ report --
print("=" * 68)
fails = 0
for name, ok, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"  -> {detail}"))
    fails += (not ok)
print("=" * 68)
print(f"  {len(results) - fails}/{len(results)} passed")
sys.exit(1 if fails else 0)
