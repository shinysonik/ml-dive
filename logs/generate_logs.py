"""fake training log with four injected anomalies, for the ml-dive demo.
writes logs/train_log.csv, logs/run_config.yaml, docs/ground_truth_debugging.json
copy only the first two into bob's workspace"""

import csv
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

LOGS = Path(__file__).parent
DOCS = LOGS.parent / "docs"
N_EPOCHS, IPE = 100, 50  # iteration and epoch are both 1-based
BASE_LR, GAMMA = 1e-3, 0.5
STEP_EPOCHS = (20, 40, 60, 80)  # lr halves at the first iteration of these epochs
SPIKE_EPOCH, SPIKE_IT, SPIKE_MULT = 21, 25, (8.0, 4.0)  # multipliers on consecutive iterations
OVERFIT_START, OVERFIT_END = 41, 60
NAN_START, NAN_END = 61, 63  # short burst, training recovers afterwards
STALL_START = 81  # loss freezes, lr stays at its last value


def first_iter(epoch): return (epoch - 1) * IPE + 1
def train_base(epoch): return 2.0 * math.exp(-0.03 * epoch)
def val_base(epoch): return 2.2 * math.exp(-0.025 * epoch)


def val_overfit(epoch):
    # val stops improving at epoch 40 and climbs linearly
    return val_base(OVERFIT_START - 1) * (1 + 0.04 * (epoch - OVERFIT_START + 1))


def lr_at(iteration):
    return BASE_LR * GAMMA ** sum(iteration >= first_iter(e) for e in STEP_EPOCHS)


def clean_losses(epoch):
    tr, va = train_base(epoch), val_base(epoch)
    if OVERFIT_START <= epoch <= OVERFIT_END:
        tr *= 1 - 0.01 * (epoch - OVERFIT_START + 1)
        va = val_overfit(epoch)
    elif epoch > OVERFIT_END:
        # the gap closes after the window
        va += (val_overfit(OVERFIT_END) - val_base(OVERFIT_END)) * 0.6 ** (epoch - OVERFIT_END)
    if epoch >= STALL_START:
        tr, va = train_base(STALL_START - 1), val_base(STALL_START - 1)
    return tr, va


def make_rows():
    rows, t0 = [], datetime(2026, 9, 25, 10, 0, 0)
    for epoch in range(1, N_EPOCHS + 1):
        for it in range(1, IPE + 1):
            iteration = first_iter(epoch) + it - 1
            tr, va = (v * (1 + random.uniform(-0.03, 0.03)) for v in clean_losses(epoch))
            if epoch == SPIKE_EPOCH and 0 <= it - SPIKE_IT < len(SPIKE_MULT):
                tr *= SPIKE_MULT[it - SPIKE_IT]
            tr, va = f"{tr:.4f}", f"{va:.4f}"
            if NAN_START <= epoch <= NAN_END:
                tr = "inf" if (epoch, it) == (NAN_START, 1) else "nan"
                va = "nan"
            rows.append({"iteration": iteration, "epoch": epoch, "train_loss": tr,
                         "val_loss": va, "lr": f"{lr_at(iteration):.6e}",
                         "timestamp": (t0 + timedelta(seconds=2 * iteration)).isoformat()})
    return rows


def write_config():
    steps = [first_iter(e) for e in STEP_EPOCHS]  # csv iteration numbers
    (LOGS / "run_config.yaml").write_text(f"""# synthetic run behind train_log.csv
MODEL: {{WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-50.pkl"}}
DATASETS: {{TRAIN: ["my_train"], TEST: ["my_val"]}}
TEST: {{EVAL_PERIOD: {IPE}}}
SOLVER:
  BASE_LR: {BASE_LR}
  STEPS: {steps}
  GAMMA: {GAMMA}
  MAX_ITER: {N_EPOCHS * IPE}
  WARMUP_ITERS: 0
  WARMUP_FACTOR: 0.001
  LR_SCHEDULER_NAME: "WarmupMultiStepLR"
  IMS_PER_BATCH: 16
  WEIGHT_DECAY: 0.0001
  CLIP_GRADIENTS: {{ENABLED: False, CLIP_TYPE: "value", CLIP_VALUE: 1.0}}
  AMP: {{ENABLED: True}}
""")


def write_truth():
    s0 = first_iter(SPIKE_EPOCH) + SPIKE_IT - 1
    spec = [  # name, detector, severity, epoch start, epoch end, iteration start, iteration end
        ("loss spike", "B", "low", SPIKE_EPOCH, SPIKE_EPOCH, s0, s0 + len(SPIKE_MULT) - 1),
        ("overfitting", "C", "medium", OVERFIT_START, OVERFIT_END, first_iter(OVERFIT_START), OVERFIT_END * IPE),
        ("non-finite loss", "A", "high", NAN_START, NAN_END, first_iter(NAN_START), NAN_END * IPE),
        ("stalled lr", "D", "medium", STALL_START, N_EPOCHS, first_iter(STALL_START), N_EPOCHS * IPE),
    ]
    keys = ("name", "detector", "severity", "epoch_start", "epoch_end", "iteration_start", "iteration_end")
    DOCS.mkdir(parents=True, exist_ok=True)
    truth = {"index_base": 1, "anomalies": [dict(zip(keys, row)) for row in spec]}
    (DOCS / "ground_truth_debugging.json").write_text(json.dumps(truth, indent=2))


def main():
    rows = make_rows()
    with (LOGS / "train_log.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    write_config()
    write_truth()
    print(f"wrote {len(rows)} rows, run_config.yaml, docs/ground_truth_debugging.json")


if __name__ == "__main__":
    main()