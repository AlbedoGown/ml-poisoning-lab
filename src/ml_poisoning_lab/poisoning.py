"""Подмена бинарных меток. Основа подготовлена agy/Sonnet, проверена и упрощена Codex."""

import math
from numbers import Integral, Real

import numpy as np


def flip_labels(
    y: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Вернуть копию меток и индексы floor(n * fraction) подмен.

    При одинаковом seed меньший бюджет — префикс того же случайного порядка.
    Входной массив и глобальный генератор случайных чисел не изменяются.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError("y должен быть numpy.ndarray")
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y должен быть непустым одномерным массивом")
    if not np.issubdtype(y.dtype, np.integer) or not np.isin(y, [0, 1]).all():
        raise ValueError("y должен содержать только целочисленные 0 и 1")
    if isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, Real):
        raise TypeError("fraction должен быть вещественным числом, не bool")
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction должен быть конечным числом от 0 до 1")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed должен быть целым числом, не bool")
    if seed < 0:
        raise ValueError("seed должен быть неотрицательным")
    count = math.floor(len(y) * fraction)
    indices = np.random.default_rng(int(seed)).permutation(len(y))[:count]
    poisoned = y.copy()
    poisoned[indices] = 1 - poisoned[indices]
    return poisoned, indices
