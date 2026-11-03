import copy

import pytest

from ml_poisoning_lab.qwen.data import build_dataset, poison_training
from ml_poisoning_lab.qwen.defenses import build_canary, data_checks, integrity_check, manifest
from ml_poisoning_lab.qwen.policy import release_decision


def good_metrics():
    return {
        "accuracy": 2 / 3,
        "invalid_rate": 0.0,
        "source_to_target_rate": 0.1,
        "per_class": {k: {"recall": 0.75, "support": 12} for k in "ABC"},
    }


def test_policy_boundaries_and_rejection_even_with_high_accuracy():
    m = good_metrics()
    assert release_decision(m) == {"accuracy_only": True, "class_aware": True, "reasons": []}
    m["accuracy"] = 0.9
    m["per_class"]["A"]["recall"] = 0
    assert release_decision(m)["accuracy_only"]
    assert not release_decision(m)["class_aware"]


@pytest.mark.parametrize(
    "key,value",
    [("accuracy", float("nan")), ("invalid_rate", True), ("source_to_target_rate", float("inf"))],
)
def test_policy_invalid(key, value):
    m = good_metrics()
    m[key] = value
    with pytest.raises(ValueError):
        release_decision(m)


def test_policy_fails_closed_on_missing_class_and_bad_support():
    m = good_metrics()
    del m["per_class"]["A"]
    with pytest.raises(ValueError):
        release_decision(m)
    m = good_metrics()
    m["per_class"]["A"]["support"] = True
    with pytest.raises(ValueError):
        release_decision(m)


def test_integrity_order_invariant_and_detects_changed_added_removed():
    rows = build_dataset()["train"]
    trusted = manifest(rows)
    assert integrity_check(rows[::-1], trusted)["accepted"]
    poisoned, _ = poison_training(rows, 0.3, 100)
    assert len(integrity_check(poisoned, trusted)["changed"]) == 54
    assert not integrity_check(rows[:-1], trusted)["accepted"]
    extra = copy.deepcopy(rows)
    extra.append({**rows[0], "id": "new"})
    assert not integrity_check(extra, trusted)["accepted"]
    with pytest.raises(ValueError):
        manifest(rows + [rows[0]])
    assert data_checks(poisoned, poisoned)["integrity"]["accepted"]


def test_canary_balanced_and_exact_text_disjoint():
    canary = build_canary()
    assert len(canary) == 36 and len({r["id"] for r in canary}) == 36
    assert [sum(r["label"] == label for r in canary) for label in "ABC"] == [12] * 3
    all_texts = {r["text"] for rows in build_dataset(1729).values() for r in rows}
    assert {r["text"] for r in canary}.isdisjoint(all_texts)
