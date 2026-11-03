"""Настоящая проверка MLX: MLX_TEST=1 pytest tests/test_qwen_mlx.py на Apple Silicon."""

import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("MLX_TEST") != "1", reason="Нужен доступ к Metal")


def test_completion_loss_excludes_prompt_and_padding():
    import mlx.core as mx
    import numpy as np

    from ml_poisoning_lab.qwen.backend import completion_loss

    batch = mx.array([[0, 0, 0, 1, 2, 0, 0]])
    lengths = mx.array([[3, 5]])
    logits = np.zeros((1, 6, 3), dtype=np.float32)
    logits[0, 2, 1] = logits[0, 3, 2] = 10

    class Model:
        def __call__(self, inputs):
            return mx.array(logits)

    loss, count = completion_loss(Model(), batch, lengths)
    assert count.item() == 2
    assert loss.item() < 0.001
    # Ошибочные предсказания в prompt/padding не должны менять loss.
    logits[0, 0, :] = [-100, 100, 100]
    logits[0, 4:, :] = [-100, 100, 100]
    changed, _ = completion_loss(Model(), batch, lengths)
    assert changed.item() == loss.item()
