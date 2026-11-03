import json
from dataclasses import replace

import numpy as np
import pytest

from ml_poisoning_lab.cli import main
from ml_poisoning_lab.experiment import Config, evaluate, make_split, run_experiment
from ml_poisoning_lab.report import write_report


@pytest.fixture(scope="module")
def result():
    return run_experiment(Config(samples=200, repeats=2, fractions=(0, 0.2, 1)))


def test_reproducible_paired_results_and_full_inversion(result):
    assert result == run_experiment(Config(samples=200, repeats=2, fractions=(0, 0.2, 1)))
    assert len(result["runs"]) == 6
    for repeat in range(2):
        baseline, partial, inverted = [r for r in result["runs"] if r["repeat"] == repeat]
        assert baseline["changed_count"] == 0
        assert partial["changed_count"] == 28
        assert inverted["changed_count"] == 140
        # При полной инверсии симметричная бинарная модель меняет ответы на противоположные.
        assert baseline["accuracy"] + inverted["accuracy"] == pytest.approx(1)
        assert sum(map(sum, baseline["confusion_matrix"])) == 60
    baseline_mean = np.mean([r["accuracy"] for r in result["runs"] if r["fraction"] == 0])
    for item in result["summary"]:
        assert item["accuracy"]["paired_delta_mean"] == pytest.approx(
            item["accuracy"]["mean"] - baseline_mean,
        )


def test_split_is_disjoint_and_has_expected_sizes():
    X_train, X_test, y_train, y_test, train, test = make_split(Config(samples=200), 42)
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(range(200))
    assert len(y_train) == len(X_train) == 140
    assert len(y_test) == len(X_test) == 60


def test_test_features_never_used_to_fit_scaler(monkeypatch):
    from ml_poisoning_lab import experiment

    original = experiment.StandardScaler.fit
    seen = []

    def spy(self, X, y=None, **kwargs):
        seen.append(X.copy())
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(experiment.StandardScaler, "fit", spy)
    X_train, X_test, y_train, y_test, *_ = make_split(Config(samples=200), 42)
    before = [a.copy() for a in (X_train, X_test, y_train, y_test)]
    evaluate(X_train, y_train, X_test, y_test)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], X_train)
    for a, b in zip(before, (X_train, X_test, y_train, y_test), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "changes",
    [
        {"samples": 99},
        {"repeats": 1},
        {"seed": -1},
        {"seed": 2**32},
        {"fractions": ()},
        {"fractions": (0.1,)},
        {"fractions": (0, 0)},
        {"fractions": (0, np.nan)},
        {"fractions": (0, 1.1)},
        {"repeats": True},
    ],
)
def test_invalid_config(changes):
    with pytest.raises(ValueError):
        replace(Config(), **changes).validate()


def test_report_artifacts_and_json_roundtrip(tmp_path, result):
    write_report(result, tmp_path)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["runs"] == result["runs"]
    page = (tmp_path / "report.html").read_text()
    assert 'lang="ru"' in page
    assert "data:image/png;base64," in page
    assert "<script" not in page
    assert (tmp_path / "poisoning.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_cli_invalid_input_has_readable_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--fractions", "0", "nan"])
    assert exc.value.code == 2
    assert "Доли подмены" in capsys.readouterr().err


def test_cli_end_to_end(tmp_path):
    assert (
        main(
            [
                "--samples",
                "100",
                "--repeats",
                "2",
                "--fractions",
                "0",
                "1",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert len(json.loads((tmp_path / "results.json").read_text())["runs"]) == 4
