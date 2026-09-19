"""Generate fake training logs with anomalies for ML-Dive demo.

Writes to logs/train_log.csv
Columns: iteration, epoch, train_loss, val_loss, lr, timestamp
"""

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

# fix seed so logs are reproducible across runs
random.seed(42)

OUT = Path(__file__).parent / "train_log.csv"
N_EPOCHS = 100
ITERS_PER_EPOCH = 50

# anomaly windows - tuned so each issue is visible but not too obvious
SPIKE_EPOCH = 21
OVERFIT_START, OVERFIT_END = 41, 60
NAN_START = 61
STALL_START = 81


def loss_curve(epoch, base=2.0, decay=0.03):
    # rough exponential decay + a bit of noise so it doesn't look fake
    v = base * math.exp(-decay * epoch)
    return max(v + random.uniform(-0.02, 0.02), 0.01)


def make_rows():
    rows = []
    t0 = datetime(2026, 9, 25, 10, 0, 0)
    lr = 1e-3

    for epoch in range(1, N_EPOCHS + 1):
        # halve lr every 20 epochs, but stop doing it once we simulate the stall
        if epoch % 20 == 0 and epoch < STALL_START:
            lr *= 0.5

        in_overfit = OVERFIT_START <= epoch <= OVERFIT_END
        is_nan = epoch >= NAN_START

        for it in range(1, ITERS_PER_EPOCH + 1):
            iteration = (epoch - 1) * ITERS_PER_EPOCH + it
            ts = t0 + timedelta(seconds=iteration * 2)

            if is_nan:
                # once NaN appears it poisons everything downstream
                tr, va = "nan", "nan"
            else:
                tr = loss_curve(epoch)
                va = loss_curve(epoch, base=2.2, decay=0.025)

                # one big spike in the middle of epoch 21
                if epoch == SPIKE_EPOCH and it == ITERS_PER_EPOCH // 2:
                    tr *= 8.0

                if in_overfit:
                # train keeps going down, val goes up
                    tr *= 0.95
                    va = va * 1.15 + 0.15

            rows.append({
                "iteration": iteration,
                "epoch": epoch,
                "train_loss": f"{tr:.4f}" if tr != "nan" else "nan",
                "val_loss": f"{va:.4f}" if va != "nan" else "nan",
                "lr": f"{lr:.6f}",
                "timestamp": ts.isoformat(),
            })

    return rows


def main():
    rows = make_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # TODO: also dump a short summary json for the demo UI
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows -> {OUT}")
    print("anomalies: spike@21, overfit@41-60, nan@61+, stalled lr@81+")


if __name__ == "__main__":
    main()