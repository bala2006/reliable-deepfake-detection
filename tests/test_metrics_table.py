import logging

import pytest

from rlroinet.train_temporal_region import (
    _best_so_far, _pct_vs_previous, _pass_fail, print_metrics_table,
)


def _entry(auc, eer, roi=None):
    entry = {
        "auc": auc, "acc": 0.5, "f1": 0.4, "precision": 0.4, "recall": 0.4,
        "eer": eer, "ece": 0.2, "region_iou": 0.4, "region_hit": 0.4,
        "fp_rate": 0.2, "fn_rate": 0.2, "confident_false_positive_rate": 0.1,
        "confident_false_negative_rate": 0.1, "review_rate": 0.3,
        "reliable_coverage": 0.7,
    }
    if roi is not None:
        entry["per_roi_iou"] = roi
    return entry


def test_best_so_far_direction(caplog):
    history = [_entry(0.70, 0.30), _entry(0.82, 0.18), _entry(0.79, 0.20)]
    assert _best_so_far(history, "auc", "higher") == pytest.approx(0.82)
    assert _best_so_far(history, "eer", "lower") == pytest.approx(0.18)
    assert _best_so_far([{"x": float("nan")}, {"x": 0.5}], "x", "higher") == pytest.approx(0.5)


def test_metrics_table_prints_all_gate_rows(caplog):
    history = [
        _entry(0.71, 0.29, roi={"PERIOCULAR": 0.5, "JAWLINE": 0.4, "MOUTH": 0.45, "HAIRLINE": 0.35}),
        _entry(0.78, 0.22, roi={"PERIOCULAR": 0.6, "JAWLINE": 0.5, "MOUTH": 0.55, "HAIRLINE": 0.4}),
    ]
    with caplog.at_level(logging.INFO):
        print_metrics_table(history, epoch=2)
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "checkpoint metrics (epoch 2)" in text
    assert "metric" in text and "previous" in text and "current" in text and "best" in text
    assert "% vs prev" in text
    for label in ("val video AUC", "accuracy", "region IoU", "region hit@0.5",
                  "false positive rate", "confident FN rate", "review rate", "reliable coverage"):
        assert label in text
    assert "0.7100" in text and "0.7800" in text
    assert "ROI PERIOCULAR" in text


def test_pct_vs_previous_direction_and_nan():
    assert _pct_vs_previous(0.50, 0.55, "higher") == " +10.0% ^".rjust(10)
    assert _pct_vs_previous(0.55, 0.50, "higher").strip().endswith("v")
    assert _pct_vs_previous(0.30, 0.25, "lower").strip().endswith("^")
    assert _pct_vs_previous(0.25, 0.30, "lower").strip().endswith("v")
    assert "n/a" in _pct_vs_previous(None, 0.5, "higher")
    assert "n/a" in _pct_vs_previous(0.5, None, "higher")
    assert "n/a" in _pct_vs_previous(0.0, 0.1, "higher")
    assert "n/a" in _pct_vs_previous(float("nan"), 0.1, "higher")


def test_metrics_table_with_single_entry_shows_blank_previous(caplog):
    with caplog.at_level(logging.INFO):
        print_metrics_table([_entry(0.75, 0.25)], epoch=1)
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "0.7500" in text


def test_pass_fail_bar():
    assert _pass_fail(0.90, "auc", "higher") == "PASS"
    assert _pass_fail(0.84, "auc", "higher") == "FAIL"
    assert _pass_fail(0.005, "confident_false_positive_rate", "lower") == "PASS"
    assert _pass_fail(0.02, "confident_false_positive_rate", "lower") == "FAIL"
    assert _pass_fail(None, "auc", "higher") == "—"
    assert _pass_fail(float("nan"), "region_iou", "higher") == "—"
    assert _pass_fail(0.5, "acc", "higher") == "—"  # no official bar -> not gated


def test_metrics_table_has_pass_column_and_overall(caplog):
    passing = _entry(0.90, 0.10, roi={"PERIOCULAR": 0.8, "JAWLINE": 0.8, "MOUTH": 0.8, "HAIRLINE": 0.8})
    passing.update({"region_iou": 0.82, "region_hit": 0.95,
                    "confident_false_positive_rate": 0.005,
                    "confident_false_negative_rate": 0.005,
                    "reliable_coverage": 0.62, "review_rate": 0.38})
    failing = dict(passing)
    failing.update({"auc": 0.80})
    with caplog.at_level(logging.INFO):
        print_metrics_table([passing, failing], epoch=2)
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "pass?" in text
    assert "acceptance bar:" in text
    assert "OVERALL acceptance" in text
    assert "PASS" in text
    assert "FAIL" in text
