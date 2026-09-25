"""reference implementation of detectors A-D from prompts/debugging.md.
usage: python logs/reference_detectors.py [path/to/train_log.csv] [--json]
prints text reports. --json also writes logs/reference_anomalies.json in the
docs/anomalies_schema.md shape (finding is auto-generated, suggested_fix is
always null: this script does not propose fixes).
if docs/ground_truth_debugging.json exists, a recall check is printed at the end."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
f4 = "{:.4f}".format
fe = "{:.4e}".format


def load(path):
    df = pd.read_csv(path, dtype=str)
    for c in ("iteration", "epoch"):
        df[c] = pd.to_numeric(df[c]).astype(int)
    for c in ("train_loss", "val_loss", "lr"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["tr_ok"] = np.isfinite(df.train_loss)
    df["va_ok"] = np.isfinite(df.val_loss)
    return df


def runs(mask):
    idx = np.flatnonzero(np.asarray(mask))
    if len(idx) == 0:
        return []
    return [(p[0], p[-1]) for p in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)]


def epoch_means(df):
    tr = df[df.tr_ok].groupby("epoch").train_loss.mean()
    va = df[df.va_ok].groupby("epoch").val_loss.mean()
    return pd.concat([tr, va], axis=1).dropna()


def detect_a(df, items):
    print("== A non-finite")
    for col, ok in (("train_loss", "tr_ok"), ("val_loss", "va_ok")):
        bad = df[~df[ok]]
        if bad.empty:
            print(f"{col}: none")
            continue
        first = bad.iloc[0]
        before = df[df.iteration < first.iteration]
        last_fin = before[before[ok]].iteration.max()
        print(f"{col}: first iteration {first.iteration}, epoch {first.epoch}, rows {len(bad)}, "
              f"epochs affected {bad.epoch.nunique()}, last finite before it: iteration {last_fin}")
    bad_any = ~(df.tr_ok & df.va_ok)
    n = 0
    for a, b in runs(bad_any):
        rec = df.iloc[b + 1].iteration if b + 1 < len(df) else None
        print(f"run: iterations {df.iteration[a]}-{df.iteration[b]}, epochs {df.epoch[a]}-{df.epoch[b]}, "
              f"recovered at iteration {rec if rec else 'never'}")
        n += 1
    tr_bad = np.flatnonzero(~df.tr_ok)
    lr_peak = slope = ratio = last_fin_it = None
    if len(tr_bad):
        i = tr_bad[0]
        lr_peak = float(df.lr[i])
        print(f"lr at first non-finite train row: {fe(df.lr[i])}")
        pre = df.iloc[:i][df.tr_ok[:i]].tail(50)
        last_fin_it = int(pre.iteration.iloc[-1]) if len(pre) else None
        if len(pre) >= 2:
            slope = float(np.polyfit(pre.iteration, pre.train_loss, 1)[0])
            ratio = float(pre.train_loss.max() / pre.train_loss.median())
            print(f"50 finite rows before: slope {fe(slope)}, max ratio to median {f4(ratio)}")
    items.clear()
    for k, (a, b) in enumerate(runs(bad_any), 1):
        rec = df.iloc[b + 1].iteration if b + 1 < len(df) else None
        items.append(dict(
            id=f"A{k}", type="nan_loss", detector="A", severity="high",
            epoch_start=int(df.epoch[a]), epoch_end=int(df.epoch[b]),
            iteration_start=int(df.iteration[a]), iteration_end=int(df.iteration[b]),
            evidence=dict(first_iteration=int(df.iteration[a]), row_count=int(b - a + 1),
                          last_finite_iteration=last_fin_it,
                          lr_at_first_nonfinite=fe(lr_peak) if lr_peak is not None else None,
                          pre_burst_slope=fe(slope) if slope is not None else None),
            finding=f"train_loss and val_loss are non-finite for iterations {df.iteration[a]}-{df.iteration[b]} "
                    f"(epochs {df.epoch[a]}-{df.epoch[b]}).",
            suggested_fix=None, fix_status="not applicable — reference detector does not propose fixes",
            source_report="reference_detectors.py"))
    return items


def detect_b(df, items):
    print("== B spikes")
    events, med = [], {}
    flagged = []
    for a, b in runs(df.tr_ok):
        v = df.train_loss.values
        for i in range(a + 10, b + 1):
            med[i] = np.median(v[i - 10:i])
            if v[i] > 3.0 * med[i]:
                flagged.append(i)
    evs = []
    for a, b in runs(np.isin(np.arange(len(df)), flagged)):
        idx = np.arange(a, b + 1)
        peak = idx[np.argmax([df.train_loss[i] / med[i] for i in idx])]
        rec = next((j for j in range(peak + 1, len(df)) if j in med and df.tr_ok[j]
                    and df.train_loss[j] <= 1.5 * med[j]), None)
        evs.append(dict(a=a, b=b, peak=peak, ratio=df.train_loss[peak] / med[peak],
                        rec=df.iteration[rec] if rec is not None else "not recovered"))
    evs.sort(key=lambda e: -e["ratio"])
    if not evs:
        print("no events")
    for e in evs[:5]:
        p = e["peak"]
        print(f"event: first iteration {df.iteration[e['a']]}, epoch {df.epoch[e['a']]}, peak iteration "
              f"{df.iteration[p]}, value {f4(df.train_loss[p])}, ratio {f4(e['ratio'])}, duration "
              f"{e['b'] - e['a'] + 1}, lr {fe(df.lr[p])}, back under 1.5x at {e['rec']}")
    for k, e in enumerate(evs[:5], 1):
        p = e["peak"]
        recovered = e["rec"] != "not recovered"
        items.append(dict(
            id=f"B{k}", type="loss_spike", detector="B", severity="low" if recovered else "medium",
            epoch_start=int(df.epoch[e["a"]]), epoch_end=int(df.epoch[e["b"]]),
            iteration_start=int(df.iteration[e["a"]]), iteration_end=int(df.iteration[e["b"]]),
            evidence=dict(peak_iteration=int(df.iteration[p]), peak_value=f4(df.train_loss[p]),
                          peak_ratio=f4(e["ratio"]), duration_iterations=int(e["b"] - e["a"] + 1),
                          lr_at_peak=fe(df.lr[p]), recovered_at=str(e["rec"])),
            finding=f"train_loss spikes to {f4(df.train_loss[p])} at iteration {df.iteration[p]} "
                    f"({f4(e['ratio'])}x the trailing median).",
            suggested_fix=None, fix_status="not applicable — reference detector does not propose fixes",
            source_report="reference_detectors.py"))
    return items


def slopes(ep, lo, hi):
    x = np.arange(lo, hi + 1)
    return np.polyfit(x, ep.train_loss[x], 1)[0], np.polyfit(x, ep.val_loss[x], 1)[0]


def detect_c(df, items):
    print("== C overfitting")
    ep = epoch_means(df)
    ep.columns = ["train_loss", "val_loss"]
    have = set(ep.index)
    wins = []
    for s in sorted(have):
        if all(e in have for e in range(s, s + 5)):
            st, sv = slopes(ep, s, s + 4)
            if st < 0 and sv > 0:
                wins.append([s, s + 4])
    merged = []
    for w in wins:
        if merged and w[0] <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], w[1])
        else:
            merged.append(list(w))
    kept, dropped = [], 0
    for lo, hi in merged:
        lo = int(ep.val_loss.loc[lo:hi].idxmin())
        tr0, tr1, va0, va1 = ep.train_loss[lo], ep.train_loss[hi], ep.val_loss[lo], ep.val_loss[hi]
        if va1 >= 1.02 * va0 and tr1 <= 0.98 * tr0:
            st, sv = slopes(ep, lo, hi)
            print(f"run: epochs {lo}-{hi}, train slope {fe(st)}, val slope {fe(sv)}, "
                  f"gap {f4(va0 - tr0)} -> {f4(va1 - tr1)}")
            kept.append((lo, hi, st, sv, va0 - tr0, va1 - tr1))
        else:
            dropped += 1
            print(f"discarded: epochs {lo}-{hi}")
    print(f"discarded runs: {dropped}")
    print(f"min val epoch {ep.val_loss.idxmin()} ({f4(ep.val_loss.min())}), "
          f"last defined epoch {ep.index.max()} val {f4(ep.val_loss.iloc[-1])}")
    for k, (lo, hi, st, sv, g0, g1) in enumerate(kept, 1):
        items.append(dict(
            id=f"C{k}", type="overfitting", detector="C", severity="medium",
            epoch_start=int(lo), epoch_end=int(hi),
            iteration_start=(lo - 1) * 50 + 1, iteration_end=hi * 50,
            evidence=dict(train_slope=fe(st), val_slope=fe(sv), gap_start=f4(g0), gap_end=f4(g1)),
            finding=f"train_loss falls while val_loss rises over epochs {lo}-{hi}; "
                    f"gap widens from {f4(g0)} to {f4(g1)}.",
            suggested_fix=None, fix_status="not applicable — reference detector does not propose fixes",
            source_report="reference_detectors.py"))
    return items


def detect_d(df, items):
    print("== D lr")
    lr = df.lr.values
    cp = np.flatnonzero(np.abs(np.diff(lr)) > 1e-9 * np.abs(lr[:-1])) + 1
    print(f"change points: {len(cp)}")
    if len(cp) <= 20:
        for i in cp:
            print(f"iteration {df.iteration[i]}, epoch {df.epoch[i]}: {fe(lr[i - 1])} -> {fe(lr[i])}")
    else:
        bounds = [0, *cp, len(df)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            print(f"segment iterations {df.iteration[a]}-{df.iteration[b - 1]}: {fe(lr[a])}")
    g = df.groupby("epoch").lr
    const = ((g.max() - g.min()) <= 1e-9 * g.max().abs())
    first = g.first()
    ep, groups, cur = epoch_means(df), [], []
    for e in first.index:
        same = cur and const[e] and abs(first[e] - first[cur[-1]]) <= 1e-9 * abs(first[e]) and e == cur[-1] + 1
        if same:
            cur.append(e)
        else:
            if cur:
                groups.append(cur)
            cur = [e] if const[e] else []
    groups.append(cur)
    stalled = []
    for r in (g_ for g_ in groups if len(g_) >= 10):
        defined = [e for e in r if e in ep.index]
        ch = lambda col: abs(ep[col][defined[-1]] / ep[col][defined[0]] - 1) if len(defined) >= 2 else None
        tc, vc = ch("train_loss"), ch("val_loss")
        label = "constant lr, loss undefined" if tc is None else ("stalled" if tc < 0.01 else "constant lr, loss moving")
        print(f"const run: epochs {r[0]}-{r[-1]}, lr {fe(first[r[0]])}, fraction of max {f4(first[r[0]] / lr.max())}, "
              f"train change {f4(tc) if tc is not None else 'n/a'}, val change {f4(vc) if vc is not None else 'n/a'}, {label}")
        if label == "stalled":
            stalled.append((r[0], r[-1], first[r[0]], first[r[0]] / lr.max(), tc, vc))
    for k, (lo, hi, lrv, frac, tc, vc) in enumerate(stalled, 1):
        items.append(dict(
            id=f"D{k}", type="stalled_lr", detector="D", severity="medium",
            epoch_start=int(lo), epoch_end=int(hi),
            iteration_start=(lo - 1) * 50 + 1, iteration_end=hi * 50,
            evidence=dict(lr_value=fe(lrv), lr_fraction_of_max=f4(frac),
                          train_loss_change=f4(tc), val_loss_change=f4(vc)),
            finding=f"lr constant at {fe(lrv)} ({f4(frac)} of max) over epochs {lo}-{hi}; "
                    f"train_loss changes by only {f4(tc)}.",
            suggested_fix=None, fix_status="not applicable — reference detector does not propose fixes",
            source_report="reference_detectors.py"))
    return items


def score(items):
    path = ROOT / "docs" / "ground_truth_debugging.json"
    if not path.exists():
        return
    truth = json.loads(path.read_text())["anomalies"]
    found = [(i["detector"], i["epoch_start"], i["epoch_end"]) for i in items]
    hit = [any(d == t["detector"] and abs(a - t["epoch_start"]) <= 1 and abs(b - t["epoch_end"]) <= 1
               for d, a, b in found) for t in truth]
    fp = [f for f in found if not any(f[0] == t["detector"] and abs(f[1] - t["epoch_start"]) <= 1
                                      and abs(f[2] - t["epoch_end"]) <= 1 for t in truth)]
    print("== score vs ground truth")
    for t, h in zip(truth, hit):
        print(f"{t['name']}: {'found' if h else 'MISSED'}")
    print(f"recall {sum(hit)}/{len(truth)}, false positives {len(fp)} {fp}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    as_json = "--json" in sys.argv
    df = load(args[0] if args else ROOT / "logs" / "train_log.csv")
    print(f"rows {len(df)}, epochs {df.epoch.min()}-{df.epoch.max()}, iterations {df.iteration.min()}-{df.iteration.max()}")
    items = []
    detect_a(df, items)
    detect_b(df, items)
    detect_c(df, items)
    detect_d(df, items)
    score(items)
    if as_json:
        out = ROOT / "logs" / "reference_anomalies.json"
        out.write_text(json.dumps(items, indent=2))
        print(f"wrote {out}")
