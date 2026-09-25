"""structural diff between two anomalies.json files (docs/anomalies_schema.md shape).
usage: python logs/score_anomalies.py <candidate.json> <reference.json>
compares detector, epoch_start/epoch_end (+-1 tolerance), severity.
never diffs finding or suggested_fix text: those are expected to differ in wording."""

import json
import sys
from pathlib import Path

TOL = 1


def load(path):
    return json.loads(Path(path).read_text())


def epoch_match(x, y):
    return abs(x["epoch_start"] - y["epoch_start"]) <= TOL and abs(x["epoch_end"] - y["epoch_end"]) <= TOL


def main():
    if len(sys.argv) != 3:
        print("usage: python logs/score_anomalies.py <candidate.json> <reference.json>")
        sys.exit(1)
    cand, ref = load(sys.argv[1]), load(sys.argv[2])

    matches = []
    unmatched_ref = list(ref)
    false_positives = []
    for c in cand:
        m = next((r for r in unmatched_ref if r["detector"] == c["detector"] and epoch_match(c, r)), None)
        if m:
            unmatched_ref.remove(m)
            matches.append((c, m))
        else:
            false_positives.append(c)

    print(f"candidate items: {len(cand)}, reference items: {len(ref)}")
    print(f"matched: {len(matches)}")
    for c, r in matches:
        sev_ok = c["severity"] == r["severity"]
        print(f"  {c['id']} <-> {r['id']}: detector {c['detector']}, epochs {c['epoch_start']}-{c['epoch_end']} "
              f"vs {r['epoch_start']}-{r['epoch_end']}, severity {c['severity']} vs {r['severity']} "
              f"{'OK' if sev_ok else 'MISMATCH'}")
    print(f"missed (in reference, not matched): {len(unmatched_ref)}")
    for r in unmatched_ref:
        print(f"  {r['id']}: detector {r['detector']}, epochs {r['epoch_start']}-{r['epoch_end']}")
    print(f"false positives (in candidate, not in reference): {len(false_positives)}")
    for c in false_positives:
        print(f"  {c['id']}: detector {c['detector']}, epochs {c['epoch_start']}-{c['epoch_end']}")

    sev_mismatches = sum(1 for c, r in matches if c["severity"] != r["severity"])
    print(f"recall {len(matches)}/{len(ref)}, false positives {len(false_positives)}, "
          f"severity mismatches {sev_mismatches}/{len(matches)}")


if __name__ == "__main__":
    main()
