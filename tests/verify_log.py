"""Prove detectron2_run.log now agrees with train_log.csv and reference_anomalies.json."""
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

log_text = (ROOT / "logs" / "detectron2_run.log").read_text(encoding="utf-8")
csv_text = (ROOT / "logs" / "train_log.csv").read_text(encoding="utf-8")
zones = json.loads((ROOT / "logs" / "reference_anomalies.json").read_text(encoding="utf-8"))

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))

# ---------------------------------------------------------------- CSV index
T0 = datetime(2026, 9, 25, 10, 0, 0)
csv_ts = {}
csv_lr = {}
for line in csv_text.splitlines()[1:]:
    p = line.split(",")
    it = int(p[0])
    csv_ts[it] = T0 + timedelta(seconds=2 * it)
    csv_lr[it] = p[4]

# ------------------------------------------------- 1. no stale contradiction
check("'iteration 600' gone", "iteration 600" not in log_text)
check("'Exiting training cleanly' gone", "Exiting training cleanly" not in log_text)
check("mentions iteration 3001 (JSON A1)", "iteration 3001" in log_text)
check("mentions iter 1025 (JSON B1)", "iter 1025" in log_text or "iter: 1025" in log_text)
check("run completes at 5000", "completed normally at iteration 5000" in log_text)

# --------------------------------------------- 2. log timestamps == CSV time
stamps = re.findall(r"\[09/25 (\d\d:\d\d:\d\d)\].*?iter: (\d+)/5000", log_text)
check("found timestamped iter lines", len(stamps) >= 6, f"only {len(stamps)}")
bad = []
for hhmmss, it in stamps:
    it = int(it)
    expected = csv_ts[it].strftime("%H:%M:%S")
    if hhmmss != expected:
        bad.append(f"iter {it}: log {hhmmss} vs csv {expected}")
check("every iter line's clock matches the CSV", not bad, "; ".join(bad))

# --------------------------------------- 3. LR values match lr_at() in CSV
lr_bad = []
for m in re.finditer(r"\[09/25 \d\d:\d\d:\d\d\][^\n]*iter: (\d+)/5000[^\n]*lr: (\S+)", log_text):
    it, logged_lr = int(m.group(1)), m.group(2)
    # compare as floats: the CSV writes %.6e, detectron2 writes %.4e — same value
    if abs(float(logged_lr) - float(csv_lr[it])) > 1e-18:
        lr_bad.append(f"iter {it}: log {logged_lr} vs csv {csv_lr[it]}")
check("every logged lr equals the CSV value", not lr_bad, "; ".join(lr_bad))

# --------------------------------------- 4. key evidence numbers from the JSON
A1 = next(z for z in zones if z["id"] == "A1")
B1 = next(z for z in zones if z["id"] == "B1")
D1 = next(z for z in zones if z["id"] == "D1")
check("A1 first_iteration 3001 present", str(A1["evidence"]["first_iteration"]) in log_text)
check("A1 window 3001-3150 present", "3001-3150" in log_text)
check("A1 lr_at_first_nonfinite present", A1["evidence"]["lr_at_first_nonfinite"] in log_text)
check("B1 peak_iteration present", str(B1["evidence"]["peak_iteration"]) in log_text)
check("B1 peak_ratio present", B1["evidence"]["peak_ratio"] in log_text)
check("D1 lr_value present", D1["evidence"]["lr_value"] in log_text)

# --------------------------------- 5. does the app's own severity rule scan pass?
sys.path.insert(0, str(ROOT))
import importlib.util
spec = importlib.util.spec_from_file_location("app", str(ROOT / "ml_dive.py"))
app = importlib.util.module_from_spec(spec)
sys.modules["app"] = app
spec.loader.exec_module(app)

counts, findings = app.scan_log("detectron2_run.log", len(log_text.encode()), log_text.encode())
print("\n  severity counts from the app's own scan_log():")
for sev, _ in app.SEVERITY_RULES:
    print(f"    {sev:<12} {counts.get(sev, 0)}")

# every zone's expected log signal must actually land INSIDE its iteration range
print("\n  position-aware cross-reference (in-range only):")
link_bad = []
for z, hits in app._build_cross_ref(zones, findings):
    sigs = app.ZONE_TYPE_LOG_SIGNALS.get(z["type"].lower(), [])
    if not sigs:
        print(f"    {z['id']} {z['type']:<12} (by design: no log signal expected)")
        continue
    ok = bool(hits)
    print(f"    {z['id']} {z['type']:<12} expects {sigs} -> {hits or 'NONE'}")
    if not ok:
        link_bad.append(z["id"])
check("every signal-bearing zone has an IN-RANGE log match", not link_bad, str(link_bad))

# the NaN evidence lines must cite the JSON's iteration range
nan_lines = [f for f in findings if f["severity"] == "NaN metric"]
check("NaN lines exist", bool(nan_lines), "none")
check(
    "a NaN line cites iteration 3001 (matches card header)",
    any("iteration 3001" in f["message"] for f in nan_lines),
    "; ".join(f["message"][:80] for f in nan_lines),
)
check(
    "no NaN line cites an iteration outside 3001-3150",
    all(
        (lambda m: not m or 3001 <= int(m) <= 3150)(re.search(r"iter(?:ation)?:? (\d+)", f["message"]).group(1) if re.search(r"iter(?:ation)?:? (\d+)", f["message"]) else None)
        for f in nan_lines
    ),
    "; ".join(f["message"][:80] for f in nan_lines),
)

# ------------------------------------------------------------------- report
print("\n" + "=" * 72)
fails = 0
for name, ok, detail in results:
    fails += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"  -> {detail}"))
print("=" * 72)
print(f"  {len(results) - fails}/{len(results)} passed")
sys.exit(1 if fails else 0)
