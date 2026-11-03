"""Метрики по основе agy/Sonnet; контракт и реализация проверены Codex."""

LABELS = ("A", "B", "C")


def parse_label(text: str) -> str | None:
    if not isinstance(text, str):
        raise TypeError("Ожидается строка")
    value = text.strip()
    return value if value in LABELS else None


def classification_metrics(truth: list[str], predictions: list[str | None]) -> dict:
    """Invalid-ответы считаются ошибками и остаются в знаменателе метрик."""
    if not truth or len(truth) != len(predictions):
        raise ValueError("Нужны непустые списки одинаковой длины")
    if any(t not in LABELS for t in truth):
        raise ValueError("Истинные метки должны быть A/B/C")
    if any(p not in (*LABELS, None) for p in predictions):
        raise ValueError("Предсказания должны быть A/B/C/None")
    cm = [[0] * 4 for _ in LABELS]
    for t, p in zip(truth, predictions, strict=True):
        cm[LABELS.index(t)][3 if p is None else LABELS.index(p)] += 1

    def divide(a, b):
        return a / b if b else 0.0

    per_class = {}
    for i, label in enumerate(LABELS):
        tp = cm[i][i]
        support = sum(cm[i])
        precision = divide(tp, sum(row[i] for row in cm))
        recall = divide(tp, support)
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": divide(2 * precision * recall, precision + recall),
        }
    return {
        "count": len(truth),
        "accuracy": sum(cm[i][i] for i in range(3)) / len(truth),
        "invalid_rate": sum(row[3] for row in cm) / len(truth),
        "macro_f1": sum(row["f1"] for row in per_class.values()) / 3,
        "confusion_matrix": cm,
        "per_class": per_class,
        "source_to_target_rate": divide(cm[0][1], sum(cm[0])),
    }
