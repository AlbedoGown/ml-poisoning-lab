"""Парные сравнения: внутри повтора меняются только обучающие метки."""

import hashlib
import platform
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .poisoning import flip_labels


@dataclass(frozen=True)
class Config:
    samples: int = 2000
    repeats: int = 5
    seed: int = 42
    fractions: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0)

    def validate(self):
        for name, value, minimum in (
            ("samples", self.samples, 100),
            ("repeats", self.repeats, 2),
            ("seed", self.seed, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name}: требуется целое число >= {minimum}")
        if self.seed + self.repeats * 3 >= 2**32:
            raise ValueError("seed слишком велик для random_state scikit-learn")
        if not self.fractions:
            raise ValueError("Нужна хотя бы одна доля подмены")
        for value in self.fractions:
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError("Доли подмены должны быть конечными числами от 0 до 1")
        if len(set(self.fractions)) != len(self.fractions):
            raise ValueError("Доли подмены не должны повторяться")
        if 0.0 not in self.fractions:
            raise ValueError("Добавьте 0: чистый baseline нужен для парного сравнения")


def digest(array):
    """Хеш массива для проверки неизменности внутри запуска."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def make_split(config, seed):
    X, y = make_classification(
        n_samples=config.samples,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        class_sep=1.3,
        flip_y=0.0,
        random_state=seed,
    )
    train, test = train_test_split(
        np.arange(config.samples),
        test_size=0.3,
        stratify=y,
        random_state=seed + 1,
    )
    return X[train], X[test], y[train], y[test], train, test


def evaluate(X_train, y_train, X_test, y_test):
    if np.unique(y_train).size != 2:
        raise ValueError("После подмены остался один класс: обучение невозможно")
    # fit видит только train; Pipeline обучает scaler на train, а не на test.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs"))
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
    }


def run_experiment(config):
    config.validate()
    rows, splits = [], []
    for repeat in range(config.repeats):
        seed = config.seed + repeat * 3
        X_train, X_test, y_train, y_test, train, test = make_split(config, seed)
        before = tuple(digest(a) for a in (X_train, X_test, y_train, y_test))
        splits.append(
            {
                "repeat": repeat,
                "data_seed": seed,
                "split_seed": seed + 1,
                "poison_seed": seed + 2,
                "train_count": len(train),
                "test_count": len(test),
                "train_indices_sha256": digest(train),
                "test_indices_sha256": digest(test),
                "clean_train_labels_sha256": digest(y_train),
                "test_labels_sha256": digest(y_test),
            }
        )
        for fraction in sorted(config.fractions):
            poisoned, indices = flip_labels(y_train, fraction, seed + 2)
            metrics = evaluate(X_train, poisoned, X_test, y_test)
            rows.append(
                {
                    "repeat": repeat,
                    "fraction": float(fraction),
                    "changed_count": len(indices),
                    "actual_fraction": len(indices) / len(y_train),
                    "changed_indices_sha256": digest(indices),
                    **metrics,
                }
            )
        after = tuple(digest(a) for a in (X_train, X_test, y_train, y_test))
        if before != after:
            raise RuntimeError("Нарушена неизменность исходных данных эксперимента")

    baseline = {row["repeat"]: row for row in rows if row["fraction"] == 0}
    summary = []
    for fraction in sorted(config.fractions):
        group = [row for row in rows if row["fraction"] == fraction]
        item = {"fraction": float(fraction)}
        for metric in ("accuracy", "balanced_accuracy", "f1"):
            values = [row[metric] for row in group]
            deltas = [row[metric] - baseline[row["repeat"]][metric] for row in group]
            item[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "paired_delta_mean": float(np.mean(deltas)),
                "paired_delta_std": float(np.std(deltas, ddof=1)),
            }
        summary.append(item)
    return {
        "schema_version": 1,
        "config": asdict(config),
        "environment": {
            "lab": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "method": {
            "attack": "random_label_flipping",
            "model": "StandardScaler + LogisticRegression",
            "test_size": 0.3,
            "features": 12,
            "informative": 6,
            "redundant": 2,
            "class_sep": 1.3,
            "generator_label_noise": 0.0,
            "std_ddof": 1,
            "positive_label": 1,
        },
        "splits": splits,
        "runs": rows,
        "summary": summary,
    }
