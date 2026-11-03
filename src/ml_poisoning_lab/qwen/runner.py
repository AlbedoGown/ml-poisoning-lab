"""Зафиксировать план, последовательно обучить адаптеры и собрать фактические результаты."""

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

from .backend import save_json
from .data import build_dataset, canonical_hash, poison_training, write_training_data

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def stage(action, model, data, output, adapter=None, config=None):
    command = [
        sys.executable,
        "-m",
        "ml_poisoning_lab.qwen.backend",
        action,
        "--model",
        model,
        "--data",
        str(data),
        "--output",
        str(output),
    ]
    if adapter:
        command += ["--adapter", str(adapter)]
    if config:
        command += ["--config", str(config)]
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    with output.with_suffix(".log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=1800
        )
    return json.loads(output.read_text())


def summarize(runs):
    metrics = ("accuracy", "macro_f1", "source_to_target_rate", "invalid_rate")
    result = {}
    for variant in ("clean", "poisoned"):
        selected = [row for row in runs if row["variant"] == variant]
        result[variant] = {
            name: {
                "mean": statistics.mean(row["metrics"][name] for row in selected),
                "std": statistics.stdev(row["metrics"][name] for row in selected),
            }
            for name in metrics
        }
    by_key = {(row["repeat"], row["variant"]): row for row in runs}
    repeats = sorted({row["repeat"] for row in runs})
    result["paired_delta_poisoned_minus_clean"] = {
        name: {
            "mean": statistics.mean(
                by_key[(i, "poisoned")]["metrics"][name] - by_key[(i, "clean")]["metrics"][name]
                for i in repeats
            ),
            "std": statistics.stdev(
                by_key[(i, "poisoned")]["metrics"][name] - by_key[(i, "clean")]["metrics"][name]
                for i in repeats
            ),
        }
        for name in metrics
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Qwen: чистое и отравленное LoRA-дообучение")
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen-v1"))
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if sys.platform != "darwin" or platform.machine() != "arm64":
        parser.error(
            "Этот backend требует Mac с Apple Silicon; исходная CPU-лаборатория доступна отдельно"
        )
    if args.iters < 1 or args.repeats < 2:
        parser.error("Нужно iters >= 1 и repeats >= 2")
    if args.output.exists():
        parser.error("Каталог уже существует: выберите новый, чтобы сохранить прежний запуск")
    os.environ.setdefault("HF_HOME", str(Path("artifacts/huggingface").resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model = snapshot_download(
        MODEL, revision=REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    args.output.mkdir(parents=True)
    dataset = build_dataset(seed=1729)
    plan = {
        "protocol": "qwen_label_flip_v3",
        "model": MODEL,
        "revision": REVISION,
        "dataset_seed": 1729,
        "train_count": len(dataset["train"]),
        "valid_count": len(dataset["valid"]),
        "test_count": len(dataset["test"]),
        "dataset_sha256": canonical_hash(dataset),
        "split_sha256": {name: canonical_hash(rows) for name, rows in dataset.items()},
        "poison_fraction_of_all_train": 0.3,
        "source": "A",
        "target": "B",
        "repeats": args.repeats,
        "training_seeds": [100 + i for i in range(args.repeats)],
        "training": {
            "loss_mask": "offset <= target_position < sequence_length",
            "iters": args.iters,
            "batch_size": 2,
            "num_layers": 8,
            "learning_rate": 2e-4,
            "max_seq_length": 256,
            "rank": 8,
            "scale": 16.0,
            "dropout": 0.0,
            "lora_keys": ["self_attn.q_proj", "self_attn.v_proj"],
        },
        "evaluation": {
            "enable_thinking": False,
            "temperature": 0,
            "max_tokens": 8,
            "strict_exact_label": True,
        },
        "source_sha256": canonical_hash(
            {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))}
        ),
    }
    # До первого обращения к тесту сохраняется полный план эксперимента.
    save_json(args.output / "plan.json", plan)
    save_json(args.output / "dataset.json", dataset)
    test_file = args.output / "test.json"
    save_json(test_file, dataset["test"])
    print("План зафиксирован. Оценивается исходная Qwen.", flush=True)
    base = stage("evaluate", model, test_file, args.output / "base.json")
    runs = []
    for repeat, seed in enumerate(plan["training_seeds"]):
        for variant in ("clean", "poisoned"):
            name = f"{repeat}-{variant}"
            directory = args.output / name
            fraction = 0.3 if variant == "poisoned" else 0.0
            train_rows, indices = poison_training(dataset["train"], fraction, seed)
            write_training_data(dataset, train_rows, tokenizer, directory / "data")
            config = {**plan["training"], "seed": seed}
            save_json(directory / "config.json", config)
            print(f"Обучение {name}: {len(indices)} подмен, {args.iters} шагов", flush=True)
            training = stage(
                "train",
                model,
                directory / "data",
                directory / "training.json",
                directory / "adapter",
                directory / "config.json",
            )
            print(f"Оценка {name}", flush=True)
            evaluation = stage(
                "evaluate", model, test_file, directory / "evaluation.json", directory / "adapter"
            )
            runs.append(
                {
                    "repeat": repeat,
                    "seed": seed,
                    "variant": variant,
                    "changed_count": len(indices),
                    "changed_indices_sha256": canonical_hash(indices),
                    "train_sha256": canonical_hash(train_rows),
                    "training": training,
                    **evaluation,
                }
            )
            print(json.dumps(evaluation["metrics"], ensure_ascii=False), flush=True)
            save_json(args.output / "progress.json", {"plan": plan, "base": base, "runs": runs})
    result = {
        "schema_version": 1,
        "plan": plan,
        "base": base,
        "runs": runs,
        "summary": summarize(runs),
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            **{
                name: importlib.metadata.version(name)
                for name in (
                    "mlx",
                    "mlx-lm",
                    "transformers",
                    "numpy",
                )
            },
        },
    }
    save_json(args.output / "results.json", result)
    print(f"Готово: {args.output / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
