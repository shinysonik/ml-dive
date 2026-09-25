"""Verify C1 + B4 + B3 + B7.

  C1 detectron2_run.log stall claim is now TRUE against train_log.csv
  B4 fix_status badge renders even when suggested_fix is null
  B3 "no log uploaded" is distinct from "log scanned, clean"
  B7 diagnosis uploader works without a metrics CSV
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "venv" / "Lib" / "site-packages"))
sys.path.insert(0, str(ROOT))

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))


import ml_dive as m  # noqa: E402
import pandas as pd  # noqa: E402

# ------------------------------------------------------------------------ C1 --
log_lines = (ROOT / "logs" / "detectron2_run.log").read_text(
    encoding="utf-8"
).splitlines()
stall = [l for l in log_lines if "schedule may have stalled" in l]
check("C1 stall line found", len(stall) == 1, f"n={len(stall)}")
stall = stall[0] if stall else ""
check("C1 false numeric claim removed", "moving < 0.001" not in stall, stall)
check("C1 still a WARNING", "WARNING" in stall, stall)
check("C1 still logs the lr value", "6.2500e-05" in stall, stall)
check("C1 ITER_RE still yields 3951 (zone 3951-5000)",
      m._line_iteration(stall) == 3951, repr(m._line_iteration(stall)))

# the claim must actually be TRUE of the shipped CSV
df = pd.read_csv(ROOT / "logs" / "train_log.csv")
win = df[df["iteration"] >= 4001]
max_delta = win["train_loss"].diff().abs().max()
check("C1 old wording would have been false (deltas > 0.001)",
      max_delta > 0.001, f"max delta={max_delta:.4f}")
first50 = win["train_loss"].head(50).mean()
last50 = win["train_loss"].tail(50).mean()
check("C1 'no trend' is TRUE (window means match)",
      abs(last50 - first50) < 0.005,
      f"first50={first50:.5f} last50={last50:.5f}")
lr_win = win["lr"].nunique()
check("C1 lr is genuinely flat across the window", lr_win == 1, f"nunique={lr_win}")

# ------------------------------------------------------------------------ B4 --
from streamlit.testing.v1 import AppTest  # noqa: E402

CSV = (ROOT / "logs" / "train_log.csv").read_bytes()
LOG = (ROOT / "logs" / "detectron2_run.log").read_bytes()
REF = (ROOT / "logs" / "reference_anomalies.json").read_bytes()
ZIP = (ROOT / "demo_repo.zip").read_bytes() if (ROOT / "demo_repo.zip").exists() else b"PK\x05\x06" + b"\x00" * 18


def find(widgets, key):
    for w in widgets:
        if getattr(w, "key", None) == key:
            return w
    return None


def all_text():
    parts = []
    for name in ("markdown", "caption", "info", "warning", "subheader", "expander"):
        for w in getattr(at, name, []):
            parts.append(str(getattr(w, "value", "")))
            parts.append(str(getattr(w, "label", "")))
    return "\n".join(parts)


def caption_text():
    return "\n".join(str(getattr(w, "value", "")) for w in getattr(at, "caption", []))


at = AppTest.from_file(str(ROOT / "ml_dive.py"), default_timeout=60)
at.run()
check("app runs at startup", not at.exception, str(at.exception))

mode = find(at.selectbox, "mode_select") or at.selectbox[0]
mode.set_value("ML Training Log Debugger")
at.run()

# --- B7 first: upload a LOG ONLY, no CSV ------------------------------------
intake = find(at.file_uploader, "ml_dive_uploader") or at.file_uploader[0]
intake.set_value([("detectron2_run.log", LOG, "text/plain")])
at.run()
check("B7 log-only intake clean", not at.exception, str(at.exception))

diag = find(at.file_uploader, "dbg_diag")
check("B7 dbg_diag uploader renders WITHOUT a csv", diag is not None,
      f"uploaders={[getattr(u, 'key', None) for u in at.file_uploader]}")

# --- B4 reference file (suggested_fix null) ---------------------------------
if diag is not None:
    diag.set_value(("reference_anomalies.json", REF, "application/json"))
    at.run()
    check("B4+B7 reference diagnosis clean", not at.exception, str(at.exception))
    txt = all_text()
    check("B4 grey badge renders with suggested_fix=null",
          ":gray-badge[" in txt, f"missing; head={txt[:300]!r}")
    check("B4 old 'Fix status:' caption fallback is gone",
          "Fix status:" not in caption_text(),
          [ln for ln in caption_text().splitlines() if "Fix status" in ln])
    check("B4 no uncoloured badge leaked for 'not applicable'",
          "[not applicable" not in txt, "plain [not applicable ...] badge found")

    # --- B7: cards must render even though there is no csv -------------------
    check("B7 diagnosis cards render without a csv", "🧠 Diagnosis" in txt,
          f"missing; info={[str(getattr(w,'value','')) for w in at.info]}")
    check("B7 no-CSV orientation line shown BEFORE the diagnosis box",
          any("Structural intake" in str(getattr(w, "value", ""))
              for w in at.info), "orientation info absent")
    # --- B3: a log WAS uploaded, so the footprint must report a real scan ----
    check("B3 with a log: footprint reports a real scan",
          "no training log uploaded" not in txt,
          "falsely claims no log despite one being uploaded")
    check("B3 with a log: in-range footprint present", "Log footprint" in txt)

# ------------------------------------------------------------------- B3 (a) --
# fresh session: CSV + diagnosis but NO log -> must admit it.
at2 = AppTest.from_file(str(ROOT / "ml_dive.py"), default_timeout=60)
at2.run()
mode2 = find(at2.selectbox, "mode_select") or at2.selectbox[0]
mode2.set_value("ML Training Log Debugger")
at2.run()
intake2 = find(at2.file_uploader, "ml_dive_uploader") or at2.file_uploader[0]
intake2.set_value([("train_log.csv", CSV, "text/csv")])
at2.run()
check("B3 no-log intake clean", not at2.exception, str(at2.exception))

diag2 = find(at2.file_uploader, "dbg_diag")
check("B3 dbg_diag present in no-log session", diag2 is not None)
if diag2 is not None:
    diag2.set_value(("reference_anomalies.json", REF, "application/json"))
    at2.run()
    check("B3 no-log diagnosis clean", not at2.exception, str(at2.exception))
    parts = []
    for name in ("markdown", "caption", "info", "warning"):
        for w in getattr(at2, name, []):
            parts.append(str(getattr(w, "value", "")))
    txt2 = "\n".join(parts)
    check("B3 no-log session ADMITS it", "no training log uploaded" in txt2,
          "honest caption absent")
    check("B3 no-log session does NOT claim a scan",
          "inside iterations 3001–3150" not in txt2
          or "No NaN warnings inside" not in txt2,
          "still asserts an empty scan happened")

# ------------------------------------------------------------------- B4 (b) --
# badge colours for the two statuses the spec names.
payload = [
    {
        "id": "T1", "type": "loss_spike", "detector": "B", "severity": "low",
        "epoch_start": 21, "epoch_end": 21,
        "iteration_start": 1025, "iteration_end": 1026,
        "evidence": {"peak_iteration": 1025},
        "finding": "spike.",
        "suggested_fix": "Enable gradient clipping.",
        "fix_status": "hypothesis — verify in repo",
        "source_report": "TRIAGE_REPORT.md",
    },
    {
        "id": "T2", "type": "nan_loss", "detector": "A", "severity": "high",
        "epoch_start": 61, "epoch_end": 63,
        "iteration_start": 3001, "iteration_end": 3150,
        "evidence": {"first_iteration": 3001},
        "finding": "nan.",
        "suggested_fix": "Lower the base lr.",
        "fix_status": "confirmed in repo (run_config.yaml:6)",
        "source_report": "TRIAGE_REPORT.md",
    },
    {
        "id": "T3", "type": "stalled_lr", "detector": "D", "severity": "medium",
        "epoch_start": 80, "epoch_end": 100,
        "iteration_start": 3951, "iteration_end": 5000,
        "evidence": {"lr_value": "6.2500e-05"},
        "finding": "stalled.",
        "suggested_fix": None,
        "fix_status": "not applicable — reference detector does not propose fixes",
        "source_report": "reference_detectors.py",
    },
]
raw3 = json.dumps(payload).encode()
at3 = AppTest.from_file(str(ROOT / "ml_dive.py"), default_timeout=60)
at3.run()
mode3 = find(at3.selectbox, "mode_select") or at3.selectbox[0]
mode3.set_value("ML Training Log Debugger")
at3.run()
intake3 = find(at3.file_uploader, "ml_dive_uploader") or at3.file_uploader[0]
intake3.set_value([("train_log.csv", CSV, "text/csv"),
                   ("detectron2_run.log", LOG, "text/plain")])
at3.run()
diag3 = find(at3.file_uploader, "dbg_diag")
check("B4 3-status fixture uploader found", diag3 is not None)
if diag3 is not None:
    diag3.set_value(("three_status.json", raw3, "application/json"))
    at3.run()
    check("B4 3-status upload clean", not at3.exception, str(at3.exception))
    parts = []
    for name in ("markdown", "caption", "info", "warning"):
        for w in getattr(at3, name, []):
            parts.append(str(getattr(w, "value", "")))
    txt3 = "\n".join(parts)
    check("B4 hypothesis -> orange badge", ":orange-badge[" in txt3, "absent")
    check("B4 confirmed -> green badge", ":green-badge[" in txt3, "absent")
    check("B4 not-applicable -> grey badge", ":gray-badge[" in txt3, "absent")
    check("B4 all three zones rendered (3 cards)", txt3.count("Log footprint") >= 3,
          f"count={txt3.count('Log footprint')}")

# ----------------------------------------- B9: fix_status is never invented --
# Two audit findings locked down:
#   (a) a MISSING fix_status used to be defaulted to "hypothesis", so the card
#       claimed a verdict the input never made;
#   (b) an unrecognised fix_status used to render as bold **[wontfix]** text
#       instead of the spec's grey badge.
payload9 = [
    {
        "id": "M1", "type": "loss_spike", "detector": "B", "severity": "low",
        "epoch_start": 21, "epoch_end": 21,
        "iteration_start": 1025, "iteration_end": 1026,
        "evidence": {"peak_iteration": 1025},
        "finding": "spike.",
        "suggested_fix": "Enable gradient clipping.",
        # fix_status deliberately omitted — the whole point of this fixture
        "source_report": "TRIAGE_REPORT.md",
    },
    {
        "id": "M2", "type": "nan_loss", "detector": "A", "severity": "high",
        "epoch_start": 61, "epoch_end": 63,
        "iteration_start": 3001, "iteration_end": 3150,
        "evidence": {"first_iteration": 3001},
        "finding": "nan.",
        "suggested_fix": "Lower the base lr.",
        "fix_status": "wontfix",
        "source_report": "TRIAGE_REPORT.md",
    },
]
raw9 = json.dumps(payload9).encode()
at9 = AppTest.from_file(str(ROOT / "ml_dive.py"), default_timeout=60)
at9.run()
mode9 = find(at9.selectbox, "mode_select") or at9.selectbox[0]
mode9.set_value("ML Training Log Debugger")
at9.run()
intake9 = find(at9.file_uploader, "ml_dive_uploader") or at9.file_uploader[0]
intake9.set_value([("train_log.csv", CSV, "text/csv"),
                   ("detectron2_run.log", LOG, "text/plain")])
at9.run()
diag9 = find(at9.file_uploader, "dbg_diag")
check("B9 uploader found with a log intake", diag9 is not None)
if diag9 is not None:
    diag9.set_value(("missing_status.json", raw9, "application/json"))
    at9.run()
    check("B9 upload clean", not at9.exception, str(at9.exception))
    parts9 = []
    for name in ("markdown", "caption", "info", "warning"):
        for w in getattr(at9, name, []):
            parts9.append(str(getattr(w, "value", "")))
    txt9 = "\n".join(parts9)
    check("B9 missing fix_status -> grey badge",
          "fix status not provided" in txt9,
          "the 'fix status not provided' badge is absent")
    check("B9 missing fix_status is NOT orange",
          ":orange-badge[" not in txt9,
          "an orange badge leaked for a status that was never supplied")
    check("B9 missing fix_status does not invent 'hypothesis'",
          "hypothesis" not in txt9, "hypothesis leaked into rendered output")
    check("B9 unrecognised fix_status ('wontfix') -> grey badge",
          ":gray-badge[" in txt9, "absent")
    check("B9 unrecognised fix_status not rendered as plain bold text",
          "**[wontfix]**" not in txt9, "plain **[wontfix]** found")
    check("B9 both cards rendered", txt9.count("Log footprint") >= 2,
          f"count={txt9.count('Log footprint')}")

# ------------------------------- misplaced training file (the reported bug) --
# Reproduces the screenshot: Structural intake holds a zip, and train_log.csv
# is dropped into the diagnosis box instead.
at4 = AppTest.from_file(str(ROOT / "ml_dive.py"), default_timeout=60)
at4.run()
mode4 = find(at4.selectbox, "mode_select") or at4.selectbox[0]
mode4.set_value("ML Training Log Debugger")
at4.run()
intake4 = find(at4.file_uploader, "ml_dive_uploader") or at4.file_uploader[0]
intake4.set_value([("demo_repo.zip", ZIP, "application/zip")])
at4.run()
check("misplaced: zip-only intake clean", not at4.exception, str(at4.exception))

diag4 = find(at4.file_uploader, "dbg_diag")
check("misplaced: diagnosis box present with no table", diag4 is not None)
if diag4 is not None:
    diag4.set_value(("train_log.csv", CSV, "text/csv"))
    at4.run()
    check("misplaced: dropping a csv does not crash", not at4.exception, str(at4.exception))
    warns = [str(getattr(w, "value", "")) for w in at4.warning]
    check("misplaced: tells the user where the csv belongs",
          any("Structural intake" in w for w in warns), f"warnings={warns}")
    check("misplaced: names the file that was misplaced",
          any("train_log.csv" in w for w in warns), f"warnings={warns}")
    check("misplaced: does NOT also claim a diagnosis parse failure",
          not any("could not be parsed" in w for w in warns), f"warnings={warns}")
    infos = [str(getattr(w, "value", "")) for w in at4.info]
    check("misplaced: orientation line still shown",
          any("Structural intake" in i for i in infos), f"infos={infos}")

# ------------------------------------------------------------------ report --
print("=" * 68)
fails = 0
for name, ok, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"  -> {detail}"))
    fails += (not ok)
print("=" * 68)
print(f"  {len(results) - fails}/{len(results)} passed")
sys.exit(1 if fails else 0)
