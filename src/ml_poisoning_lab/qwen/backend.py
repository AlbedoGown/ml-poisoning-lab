"""Один MLX-процесс на этап: память и состояние адаптера не переходят между этапами."""

import argparse
import json
import time
from pathlib import Path

from .data import messages
from .metrics import classification_metrics, parse_label


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def tokenize_training(path, tokenizer, max_length):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    prepared = []
    for row in rows:
        prefix = tokenizer.encode(row["prompt"], add_special_tokens=False)
        answer = tokenizer.encode(row["completion"], add_special_tokens=False)
        tokens = prefix + answer + [tokenizer.eos_token_id]
        if not prefix or not answer or len(tokens) > max_length:
            raise ValueError("Пустой пример или превышен max_seq_length: обрезка запрещена")
        # MLX loss mask: targets at positions >= offset; первый ответ находится len(prefix).
        prepared.append((tokens, len(prefix)))
    return prepared


def completion_loss(model, batch, lengths):
    """Только ответ и EOS: строгая верхняя граница исключает padding после EOS."""
    import mlx.core as mx
    import mlx.nn as nn

    targets = batch[:, 1:]
    logits = model(batch[:, :-1])
    positions = mx.arange(1, targets.shape[1] + 1)
    mask = (positions >= lengths[:, 0:1]) & (positions < lengths[:, 1:])
    count = mask.sum()
    losses = nn.losses.cross_entropy(logits, targets)
    return (losses * mask).astype(mx.float32).sum() / count, count


def train_adapter(args, model, tokenizer, config):
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers

    train_set = tokenize_training(args.data / "train.jsonl", tokenizer, config["max_seq_length"])
    valid_set = tokenize_training(args.data / "valid.jsonl", tokenizer, config["max_seq_length"])
    mx.random.seed(config["seed"])
    np.random.seed(config["seed"])
    model.freeze()
    lora = {
        "rank": 8,
        "scale": 16.0,
        "dropout": 0.0,
        "keys": ["self_attn.q_proj", "self_attn.v_proj"],
    }
    linear_to_lora_layers(model, config["num_layers"], lora)
    parameters = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    if not parameters:
        raise RuntimeError("Не найдено обучаемых LoRA-параметров")
    args.adapter.mkdir(parents=True, exist_ok=True)
    save_json(
        args.adapter / "adapter_config.json",
        {
            "fine_tune_type": "lora",
            "num_layers": config["num_layers"],
            "lora_parameters": lora,
        },
    )
    history = {"train": [], "validation": []}

    class Callback(TrainingCallback):
        def on_train_loss_report(self, info):
            history["train"].append(info)
            print(json.dumps({"training": info}), flush=True)

        def on_val_loss_report(self, info):
            history["validation"].append(info)

    start = time.monotonic()
    train(
        model,
        optim.Adam(learning_rate=config["learning_rate"]),
        train_set,
        valid_set,
        args=TrainingArgs(
            batch_size=config["batch_size"],
            iters=config["iters"],
            val_batches=-1,
            steps_per_report=10,
            steps_per_eval=config["iters"],
            steps_per_save=config["iters"],
            max_seq_length=config["max_seq_length"],
            adapter_file=args.adapter / "adapters.safetensors",
        ),
        training_callback=Callback(),
        loss=completion_loss,
    )
    save_json(
        args.output,
        {
            "config": config,
            "trainable_parameters": parameters,
            "duration_seconds": time.monotonic() - start,
            "history": history,
        },
    )


def evaluate_model(args, model, tokenizer):
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    model.eval()
    dataset = json.loads(args.data.read_text())
    predictions = []
    start = time.monotonic()
    for i, row in enumerate(dataset):
        prompt = tokenizer.apply_chat_template(
            messages(row),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=8,
            sampler=make_sampler(temp=0),
            verbose=False,
        )
        predictions.append(
            {"id": row["id"], "truth": row["label"], "prediction": parse_label(raw), "raw": raw}
        )
        if (i + 1) % 9 == 0:
            print(f"Оценено {i + 1}/{len(dataset)}", flush=True)
        mx.clear_cache()
    metrics = classification_metrics(
        [row["truth"] for row in predictions],
        [row["prediction"] for row in predictions],
    )
    save_json(
        args.output,
        {
            "metrics": metrics,
            "predictions": predictions,
            "duration_seconds": time.monotonic() - start,
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    from mlx_lm import load

    model, tokenizer = load(
        args.model,
        adapter_path=(str(args.adapter) if args.action == "evaluate" and args.adapter else None),
        trust_remote_code=False,
    )
    if args.action == "train":
        train_adapter(args, model, tokenizer, json.loads(args.config.read_text()))
    else:
        evaluate_model(args, model, tokenizer)


if __name__ == "__main__":
    main()
