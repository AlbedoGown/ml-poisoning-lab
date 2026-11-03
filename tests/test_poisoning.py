import numpy as np
import pytest

from ml_poisoning_lab.poisoning import flip_labels


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint8])
@pytest.mark.parametrize("fraction", [0, 0.09, 0.1, 0.5, 1])
def test_budget_and_only_selected_labels_change(dtype, fraction):
    y = np.array([0, 1] * 5, dtype=dtype)
    original = y.copy()
    poisoned, indices = flip_labels(y, fraction, 42)
    assert len(indices) == int(len(y) * fraction)
    assert len(np.unique(indices)) == len(indices)
    assert poisoned.dtype == y.dtype
    assert not np.shares_memory(y, poisoned)
    np.testing.assert_array_equal(y, original)
    np.testing.assert_array_equal(np.flatnonzero(y != poisoned), np.sort(indices))
    np.testing.assert_array_equal(poisoned[indices], 1 - y[indices])


def test_repeatability_nested_budgets_and_independent_rng():
    y = np.array([0, 1] * 50)
    state = np.random.get_state()
    small, a = flip_labels(y, 0.1, 12)
    large, b = flip_labels(y, 0.5, 12)
    again, c = flip_labels(y, 0.1, 12)
    np.testing.assert_array_equal(small, again)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(a, b[: len(a)])
    assert set(a) <= set(b)
    assert not np.array_equal(small, large)
    new_state = np.random.get_state()
    assert state[0] == new_state[0]
    np.testing.assert_array_equal(state[1], new_state[1])
    assert state[2:] == new_state[2:]


@pytest.mark.parametrize(
    "y",
    [
        [0, 1],
        np.array([]),
        np.array([[0, 1]]),
        np.array([0.0, 1.0]),
        np.array([False, True]),
        np.array([0, 2]),
        np.array([-1, 1]),
        np.array(["0", "1"]),
    ],
)
def test_invalid_labels(y):
    with pytest.raises((ValueError, TypeError)):
        flip_labels(y, 0.2, 42)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, np.nan, np.inf, -np.inf, True, "0.2", None])
def test_invalid_fractions(fraction):
    with pytest.raises((ValueError, TypeError)):
        flip_labels(np.array([0, 1]), fraction, 42)


@pytest.mark.parametrize("seed", [-1, 1.5, True, np.bool_(True), "42", None])
def test_invalid_seeds(seed):
    with pytest.raises((ValueError, TypeError)):
        flip_labels(np.array([0, 1]), 0.2, seed)


def test_single_class_and_numpy_scalar_arguments():
    y = np.zeros(10, dtype=np.int8)
    result, indices = flip_labels(y, np.float32(0.5), np.int64(2))
    assert len(indices) == 5
    assert result.sum() == 5
