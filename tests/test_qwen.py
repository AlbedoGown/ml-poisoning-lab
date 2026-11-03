import copy

import pytest
from sklearn.metrics import accuracy_score, f1_score

from ml_poisoning_lab.qwen.backend import tokenize_training
from ml_poisoning_lab.qwen.data import build_dataset, canonical_hash, poison_training
from ml_poisoning_lab.qwen.metrics import classification_metrics, parse_label
from ml_poisoning_lab.qwen.runner import summarize


def test_templates_and_texts_are_disjoint_and_repeatable():
    dataset = build_dataset()
    assert dataset == build_dataset()
    assert canonical_hash(dataset) != canonical_hash(build_dataset(43))
    assert [len(dataset[s]) for s in ("train", "valid", "test")] == [180, 36, 54]
    for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
        for key in ("id", "template_id", "text"):
            assert {r[key] for r in dataset[first]}.isdisjoint(r[key] for r in dataset[second])


def test_targeted_poisoning_does_not_change_text_or_other_splits():
    dataset = build_dataset()
    before = copy.deepcopy(dataset)
    poisoned, indices = poison_training(dataset["train"], 0.3, 42)
    assert dataset == before
    assert len(indices) == 54
    assert len(set(indices)) == 54
    for i, (old, new) in enumerate(zip(dataset["train"], poisoned, strict=True)):
        if i in indices:
            assert old["label"] == "A" and new["label"] == "B"
        else:
            assert old == new
        assert old["text"] == new["text"]
    small, a = poison_training(dataset["train"], 0.1, 42)
    assert a == indices[:18]
    assert small == poison_training(dataset["train"], 0.1, 42)[0]
    with pytest.raises(ValueError):
        poison_training(dataset["train"], 0.4, 42)


@pytest.mark.parametrize(
    "text,expected",
    [(" A\n", "A"), ("B", "B"), ("C.", None), ("<think>A</think>", None), ("A or B", None)],
)
def test_strict_parse(text, expected):
    assert parse_label(text) == expected


def test_metrics_include_invalid_and_agree_with_sklearn():
    truth = ["A", "A", "A", "B", "B", "C"]
    prediction = ["A", "B", None, "B", "C", "C"]
    result = classification_metrics(truth, prediction)
    assert result["accuracy"] == 0.5
    assert result["invalid_rate"] == 1 / 6
    assert result["source_to_target_rate"] == 1 / 3
    assert result["confusion_matrix"] == [[1, 1, 0, 1], [0, 1, 1, 0], [0, 0, 1, 0]]
    mapped = [p or "invalid" for p in prediction]
    assert result["accuracy"] == accuracy_score(truth, mapped)
    assert result["macro_f1"] == pytest.approx(
        f1_score(truth, mapped, labels=["A", "B", "C"], average="macro", zero_division=0),
    )
    absent = classification_metrics(["C"], [None])
    assert absent["source_to_target_rate"] == absent["macro_f1"] == 0


@pytest.mark.parametrize("truth,pred", [([], []), (["A"], []), (["X"], ["A"]), (["A"], ["X"])])
def test_metrics_invalid_input(truth, pred):
    with pytest.raises(ValueError):
        classification_metrics(truth, pred)


def test_response_mask_and_no_silent_truncation(tmp_path):
    class Tokenizer:
        eos_token_id = 99

        def encode(self, text, **kwargs):
            return list(text.encode())

    file = tmp_path / "train.jsonl"
    file.write_text('{"prompt":"abc","completion":"A"}\n')
    assert tokenize_training(file, Tokenizer(), 5) == [([97, 98, 99, 65, 99], 3)]
    with pytest.raises(ValueError, match="обрезка"):
        tokenize_training(file, Tokenizer(), 4)


def test_summary_uses_paired_deltas_and_sample_std():
    runs = []
    for repeat, (clean, poisoned) in enumerate(((0.9, 0.6), (0.7, 0.6), (0.8, 0.6))):
        for variant, accuracy in (("clean", clean), ("poisoned", poisoned)):
            runs.append(
                {
                    "repeat": repeat,
                    "variant": variant,
                    "metrics": {
                        "accuracy": accuracy,
                        "macro_f1": accuracy,
                        "source_to_target_rate": 1 - accuracy,
                        "invalid_rate": 0,
                    },
                }
            )
    result = summarize(runs)
    assert result["clean"]["accuracy"]["mean"] == pytest.approx(0.8)
    assert result["clean"]["accuracy"]["std"] == pytest.approx(0.1)
    assert result["paired_delta_poisoned_minus_clean"]["accuracy"]["mean"] == pytest.approx(-0.2)
    assert result["paired_delta_poisoned_minus_clean"]["accuracy"]["std"] == pytest.approx(0.1)
