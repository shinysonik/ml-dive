"""Run every verification script and print one summary line each.

    python tests/run_all.py

Each script boots the real app headlessly (Streamlit's AppTest), uploads the
actual demo files, and asserts specific behaviour.  They are the only thing
proving ml_dive.py still works after a change, so run this before committing.

Exit code is 0 only when every script passes.
"""
import re
import subprocess
import sys
from pathlib import Path

TESTS = [
    "verify_position.py",   # 38  log line <-> anomaly zone cross-referencing
    "verify_fixes.py",      # 12  shared diagnosis upload + non-finite audit
    "verify_log.py",        # 18  text log agrees with the metric CSV
    "verify_minor.py",      # 20  earlier small fixes stay fixed
    "verify_c1b4b3b7.py",   # 37  badges, stall claim, no-log state, csv gate
]

HERE = Path(__file__).resolve().parent
PY = sys.executable

print(f"python : {PY}")
print(f"tests  : {HERE}")
print("=" * 72)

failed = []
got_total = 0
for name in TESTS:
    script = HERE / name
    if not script.exists():
        print(f"  [MISSING] {name}")
        failed.append(name)
        continue
    proc = subprocess.run(
        [PY, str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    match = re.search(r"(\d+)/(\d+) passed", out)
    if match:
        good, total = int(match.group(1)), int(match.group(2))
        got_total += total
        ok = proc.returncode == 0 and good == total
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<24} {good}/{total}")
        if not ok:
            failed.append(name)
            # show the failing detail rather than making people dig for it
            for line in out.splitlines():
                if "[FAIL]" in line:
                    print(f"           {line.strip()}")
    else:
        print(f"  [FAIL] {name:<24} no result line (exit {proc.returncode})")
        failed.append(name)
        for line in out.splitlines()[-15:]:
            if line.strip():
                print(f"           {line}")

print("=" * 72)
if failed:
    print(f"  {len(failed)} of {len(TESTS)} scripts failed: {', '.join(failed)}")
    sys.exit(1)
print(f"  all {len(TESTS)} scripts passed ({got_total} checks)")
