"""reference implementation of detectors A-D from prompts/debugging.md.
usage: python logs/reference_detectors.py [path/to/train_log.csv]
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


def detect_a(df):
    print("== A non-finite")
    found = []
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
    for a, b in runs(bad_any):
        rec = df.iloc[b + 1].iteration if b + 1 < len(df) else None
        print(f"run: iterations {df.iteration[a]}-{df.iteration[b]}, epochs {df.epoch[a]}-{df.epoch[b]}, "
              f"recovered at iteration {rec if rec else 'never'}")
        found.append(("A", df.epoch[a], df.epoch[b]))
    tr_bad = np.flatnonzero(~df.tr_ok)
    if len(tr_bad):
        i = tr_bad[0]
        print(f"lr at first non-finite train row: {fe(df.lr[i])}")
        pre = df.iloc[:i][df.tr_ok[:i]].tail(50)
        if len(pre) >= 2:
            slope = np.polyfit(pre.iteration, pre.train_loss, 1)[0]
            print(f"50 finite rows before: slope {fe(slope)}, max ratio to median "
                  f"{f4(pre.train_loss.max() / pre.train_loss.median())}")
    return found


def detect_b(df):
    print("== B spikes")
    events, med = [], {}
    flagged = []
    for a, b in runs(df.tr_ok):
        v = df.train_loss.values
        for i in range(a + 10, b + 1):
            med[i] = np.median(v[i - 10:i])
            if v[i] > 3.0 * med[i]:
                flagged.append(i)
    for a, b in runs(np.isin(np.arange(len(df)), flagged)):
        idx = np.arange(a, b + 1)
        peak = idx[np.argmax([df.train_loss[i] / med[i] for i in idx])]
        rec = next((j for j in range(peak + 1, len(df)) if j in med and df.tr_ok[j]
                    and df.train_loss[j] <= 1.5 * med[j]), None)
        events.append(dict(a=a, b=b, peak=peak, ratio=df.train_loss[peak] / med[peak],
                           rec=df.iteration[rec] if rec is not None else "not recovered"))
    events.sort(key=lambda e: -e["ratio"])
    found = []
    for e in events[:5]:
        p = e["peak"]
        print(f"event: first iteration {df.iteration[e['a']]}, epoch {df.epoch[e['a']]}, peak iteration "
              f"{df.iteration[p]}, value {f4(df.train_loss[p])}, ratio {f4(e['ratio'])}, duration "
              f"{e['b'] - e['a'] + 1}, lr {fe(df.lr[p])}, back under 1.5x at {e['rec']}")
        found.append(("B", df.epoch[e["a"]], df.epoch[e["b"]]))
    if not events:
        print("no events")
    return found


def slopes(ep, lo, hi):
    x = np.arange(lo, hi + 1)
    return np.polyfit(x, ep.train_loss[x], 1)[0], np.polyfit(x, ep.val_loss[x], 1)[0]


def detect_c(df):
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
    found, dropped = [], 0
    for lo, hi in merged:
        lo = int(ep.val_loss.loc[lo:hi].idxmin())  # a run starts where val is best
        tr0, tr1, va0, va1 = ep.train_loss[lo], ep.train_loss[hi], ep.val_loss[lo], ep.val_loss[hi]
        if va1 >= 1.02 * va0 and tr1 <= 0.98 * tr0:
            st, sv = slopes(ep, lo, hi)
            print(f"run: epochs {lo}-{hi}, train slope {fe(st)}, val slope {fe(sv)}, "
                  f"gap {f4(va0 - tr0)} -> {f4(va1 - tr1)}")
            found.append(("C", lo, hi))
        else:
            dropped += 1
            print(f"discarded: epochs {lo}-{hi}")
    print(f"discarded runs: {dropped}")
    print(f"min val epoch {ep.val_loss.idxmin()} ({f4(ep.val_loss.min())}), "
          f"last defined epoch {ep.index.max()} val {f4(ep.val_loss.iloc[-1])}")
    return found


def detect_d(df):
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
    found = []
    for r in (g_ for g_ in groups if len(g_) >= 10):
        defined = [e for e in r if e in ep.index]
        ch = lambda col: abs(ep[col][defined[-1]] / ep[col][defined[0]] - 1) if len(defined) >= 2 else None
        tc, vc = ch("train_loss"), ch("val_loss")
        label = "constant lr, loss undefined" if tc is None else ("stalled" if tc < 0.01 else "constant lr, loss moving")
        print(f"const run: epochs {r[0]}-{r[-1]}, lr {fe(first[r[0]])}, fraction of max {f4(first[r[0]] / lr.max())}, "
              f"train change {f4(tc) if tc is not None else 'n/a'}, val change {f4(vc) if vc is not None else 'n/a'}, {label}")
        if label == "stalled":
            found.append(("D", r[0], r[-1]))
    return found


def score(found):
    path = ROOT / "docs" / "ground_truth_debugging.json"
    if not path.exists():
        return
    truth = json.loads(path.read_text())["anomalies"]
    hit = [any(d == t["detector"] and abs(a - t["epoch_start"]) <= 1 and abs(b - t["epoch_end"]) <= 1
               for d, a, b in found) for t in truth]
    fp = [f for f in found if not any(f[0] == t["detector"] and abs(f[1] - t["epoch_start"]) <= 1
                                      and abs(f[2] - t["epoch_end"]) <= 1 for t in truth)]
    print("== score vs ground truth")
    for t, h in zip(truth, hit):
        print(f"{t['name']}: {'found' if h else 'MISSED'}")
    print(f"recall {sum(hit)}/{len(truth)}, false positives {len(fp)} {fp}")


if __name__ == "__main__":
    df = load(sys.argv[1] if len(sys.argv) > 1 else ROOT / "logs" / "train_log.csv")
    print(f"rows {len(df)}, epochs {df.epoch.min()}-{df.epoch.max()}, iterations {df.iteration.min()}-{df.iteration.max()}")
    score(detect_a(df) + detect_b(df) + detect_c(df) + detect_d(df))
