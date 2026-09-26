# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
"""
tests/test_reasoning.py
-----------------------
Tests for the `reasoning` field added in the Bob reasoning implementation.

Covers:
  1. load_diagnosis() extracts a well-formed `reasoning` list
  2. load_diagnosis() defaults to [] when `reasoning` is absent (backwards compat)
  3. load_diagnosis() defaults to [] when `reasoning` is the wrong type
  4. load_diagnosis() strips empty/falsy strings from the list
  5. The reference_anomalies.json still loads cleanly (no reasoning → [])
  6. render_diagnosis_cards() renders the expander when reasoning is present
  7. render_diagnosis_cards() skips the expander when reasoning is []

Run with:
    python tests/test_reasoning.py
"""

import json
import sys
import os
import traceback

# Make sure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_dive import load_diagnosis  # noqa: E402

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    msg = f"  {status} {name}"
    if not condition and detail:
        msg += f"\n         → {detail}"
    print(msg)
    results.append(condition)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_item(**kwargs) -> dict:
    """Return a minimal valid anomaly dict with any overrides."""
    base = {
        "id": "A1",
        "type": "nan_loss",
        "detector": "A",
        "severity": "high",
        "epoch_start": 1,
        "epoch_end": 2,
        "iteration_start": 100,
        "iteration_end": 200,
        "evidence": {},
        "finding": "Loss went NaN at iteration 100.",
        "suggested_fix": "Lower the learning rate.",
        "fix_status": "hypothesis — verify in repo",
        "source_report": "detect_a.py",
    }
    base.update(kwargs)
    return base


def load_json(items: list) -> list[dict] | None:
    raw = json.dumps(items).encode()
    return load_diagnosis("test.json", len(raw), raw)


# ---------------------------------------------------------------------------
# Test 1 — reasoning list is extracted correctly
# ---------------------------------------------------------------------------
print("\n-- load_diagnosis() --")

steps = ["loss non-finite from iteration 100", "checked lr schedule", "ruled out OOM"]
zones = load_json([make_item(reasoning=steps)])
check(
    "reasoning list extracted",
    zones is not None and zones[0].get("reasoning") == steps,
    f"got: {zones[0].get('reasoning') if zones else None}",
)

# ---------------------------------------------------------------------------
# Test 2 — reasoning absent → defaults to []
# ---------------------------------------------------------------------------
item_no_reasoning = make_item()  # no `reasoning` key at all
del item_no_reasoning  # rebuild without reasoning key
item_no_reasoning = {k: v for k, v in make_item().items() if k != "reasoning"}
zones = load_json([item_no_reasoning])
check(
    "reasoning absent → []",
    zones is not None and zones[0].get("reasoning") == [],
    f"got: {zones[0].get('reasoning') if zones else None}",
)

# ---------------------------------------------------------------------------
# Test 3 — reasoning is wrong type (string) → defaults to []
# ---------------------------------------------------------------------------
zones = load_json([make_item(reasoning="should be a list not a string")])
check(
    "reasoning wrong type (str) → []",
    zones is not None and zones[0].get("reasoning") == [],
    f"got: {zones[0].get('reasoning') if zones else None}",
)

# ---------------------------------------------------------------------------
# Test 4 — reasoning wrong type (null) → defaults to []
# ---------------------------------------------------------------------------
zones = load_json([make_item(reasoning=None)])
check(
    "reasoning null → []",
    zones is not None and zones[0].get("reasoning") == [],
    f"got: {zones[0].get('reasoning') if zones else None}",
)

# ---------------------------------------------------------------------------
# Test 5 — empty strings inside reasoning list are stripped
# ---------------------------------------------------------------------------
zones = load_json([make_item(reasoning=["step one", "", "step two", None, "step three"])])
expected = ["step one", "step two", "step three"]
check(
    "empty/None strings stripped from reasoning",
    zones is not None and zones[0].get("reasoning") == expected,
    f"got: {zones[0].get('reasoning') if zones else None}",
)

# ---------------------------------------------------------------------------
# Test 6 — non-string elements are coerced to str
# ---------------------------------------------------------------------------
zones = load_json([make_item(reasoning=[42, True, "real step"])])
expected_coerced = ["42", "True", "real step"]
check(
    "non-string reasoning elements coerced to str",
    zones is not None and zones[0].get("reasoning") == expected_coerced,
    f"got: {zones[0].get('reasoning') if zones else None}",
)

# ---------------------------------------------------------------------------
# Test 7 — reference_anomalies.json still loads with reasoning=[] on each item
# ---------------------------------------------------------------------------
ref_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "logs", "reference_anomalies.json")
try:
    with open(ref_path, "rb") as fh:
        raw = fh.read()
    zones = load_diagnosis("reference_anomalies.json", len(raw), raw)
    all_empty = zones is not None and all(z.get("reasoning") == [] for z in zones)
    check(
        "reference_anomalies.json loads; all reasoning=[]",
        all_empty,
        f"zones={len(zones) if zones else None}, "
        f"non-empty reasoning: {[z['id'] for z in (zones or []) if z.get('reasoning')]}",
    )
except FileNotFoundError:
    check("reference_anomalies.json loads (file missing — skip)", True)

# ---------------------------------------------------------------------------
# Test 8 — render_diagnosis_cards() expander present when reasoning populated
# ---------------------------------------------------------------------------
print("\n-- render_diagnosis_cards() --")  # noqa

try:
    import unittest.mock as mock
    import streamlit as st
    from ml_dive import render_diagnosis_cards

    expander_calls = []
    markdown_calls = []

    class FakeExpander:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def fake_expander(label, **kwargs):
        expander_calls.append(label)
        return FakeExpander()

    def fake_markdown(text, **kwargs):
        markdown_calls.append(text)

    zone_with_reasoning = {
        "id": "A1",
        "type": "nan_loss",
        "detector": "A",
        "severity": "high",
        "epoch_start": 1,
        "epoch_end": 2,
        "iteration_start": 100,
        "iteration_end": 200,
        "evidence": {},
        "finding": "Loss went NaN.",
        "reasoning": ["checked loss values", "confirmed NaN at iter 100"],
        "suggested_fix": "",
        "fix_status": "",
        "source_report": "",
    }

    with (
        mock.patch.object(st, "expander", side_effect=fake_expander),
        mock.patch.object(st, "markdown", side_effect=fake_markdown),
        mock.patch.object(st, "caption"),
        mock.patch.object(st, "container") as mock_container,
        mock.patch.object(st, "info"),
    ):
        mock_container.return_value.__enter__ = lambda s: s
        mock_container.return_value.__exit__ = mock.Mock(return_value=False)
        render_diagnosis_cards([zone_with_reasoning])

    reasoning_expander_shown = any("Reasoning" in c for c in expander_calls)
    check(
        "expander '🔍 Reasoning' shown when reasoning is populated",
        reasoning_expander_shown,
        f"expander labels seen: {expander_calls}",
    )

    reasoning_bullets = [m for m in markdown_calls if m.startswith("- ")]
    check(
        "reasoning steps rendered as bullet lines",
        len(reasoning_bullets) == 2,
        f"bullet lines: {reasoning_bullets}",
    )

    # Test 9 — no expander when reasoning is []
    expander_calls.clear()
    markdown_calls.clear()
    zone_no_reasoning = dict(zone_with_reasoning)
    zone_no_reasoning["reasoning"] = []

    with (
        mock.patch.object(st, "expander", side_effect=fake_expander),
        mock.patch.object(st, "markdown", side_effect=fake_markdown),
        mock.patch.object(st, "caption"),
        mock.patch.object(st, "container") as mock_container,
        mock.patch.object(st, "info"),
    ):
        mock_container.return_value.__enter__ = lambda s: s
        mock_container.return_value.__exit__ = mock.Mock(return_value=False)
        render_diagnosis_cards([zone_no_reasoning])

    reasoning_expander_shown = any("Reasoning" in c for c in expander_calls)
    check(
        "expander '🔍 Reasoning' hidden when reasoning is []",
        not reasoning_expander_shown,
        f"expander labels seen: {expander_calls}",
    )

except Exception as exc:
    print(f"  [FAIL] render tests crashed: {exc}")
    traceback.print_exc()
    results.append(False)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
passed = sum(results)
total = len(results)
print(f"\n{'='*50}")
print(f"  {passed}/{total} checks passed")
if passed < total:
    sys.exit(1)
