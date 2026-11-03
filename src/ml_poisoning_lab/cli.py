"""Запуск локального эксперимента и создание отчёта."""

import argparse
from pathlib import Path

from .experiment import Config, run_experiment
from .report import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="ML Poisoning Lab: подмена обучающих меток")
    parser.add_argument("--samples", type=int, default=2000, help="Число примеров, минимум 100")
    parser.add_argument("--repeats", type=int, default=5, help="Число парных повторов, минимум 2")
    parser.add_argument("--seed", type=int, default=42, help="Начальное зерно случайности")
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 1]
    )
    parser.add_argument("--output", type=Path, default=Path("results"), help="Каталог результатов")
    args = parser.parse_args(argv)
    config = Config(args.samples, args.repeats, args.seed, tuple(args.fractions))
    try:
        result = run_experiment(config)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        write_report(result, args.output)
    except OSError as exc:
        parser.exit(1, f"Не удалось записать отчёт: {exc}\n")
    print("Подмена | Accuracy ± SD | Изменение к baseline, п.п.")
    for row in result["summary"]:
        metric = row["accuracy"]
        print(
            f"{row['fraction']:7.0%} | {metric['mean']:.3f} ± {metric['std']:.3f}"
            f" | {metric['paired_delta_mean'] * 100:+.2f}"
        )
    print(f"Отчёт: {(args.output / 'report.html').resolve()}")
    return 0
