"""Лабораторное правило допуска по основе agy, проверенной и адаптированной Codex."""

import math


def release_decision(metrics: dict) -> dict:
    def rate(value):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Метрика должна быть конечным числом от 0 до 1")
        return value

    try:
        accuracy = rate(metrics["accuracy"])
        invalid = rate(metrics["invalid_rate"])
        target = rate(metrics["source_to_target_rate"])
        recalls = {}
        for label in ("A", "B", "C"):
            row = metrics["per_class"][label]
            recalls[label] = rate(row["recall"])
            support = row["support"]
            if isinstance(support, bool) or not isinstance(support, int) or support < 1:
                raise ValueError("Нужны положительные целые support")
    except (KeyError, TypeError) as exc:
        raise ValueError("Неполные или неверно структурированные метрики") from exc
    reasons = []
    if accuracy < 2 / 3:
        reasons.append("Общая точность ниже 2/3")
    if invalid != 0:
        reasons.append("Есть ответы неверного формата")
    if target > 0.1:
        reasons.append("Доля переходов A→B выше 10%")
    for label, recall in recalls.items():
        if recall < 0.75:
            reasons.append(f"Recall класса {label} ниже 75%")
    return {"accuracy_only": accuracy >= 2 / 3, "class_aware": not reasons, "reasons": reasons}
