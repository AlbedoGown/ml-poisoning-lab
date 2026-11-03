"""Публично пригодные артефакты из результатов; веса и локальные пути не копируются."""

import argparse
import base64
import html
import io
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .backend import save_json


def write_report(result, output):
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "results.json", result)
    figure = Figure(figsize=(11, 4.5), layout="constrained", facecolor="#f8fafc")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    names = ("Исходная Qwen", "Чистый LoRA", "Отравленный LoRA")
    for ax, metric, title in (
        (axes[0], "accuracy", "Правильные ответы: выше лучше"),
        (axes[1], "source_to_target_rate", "Ошибки A → B: ниже лучше"),
    ):
        means = [result["base"]["metrics"][metric]] + [
            result["summary"][variant][metric]["mean"] for variant in ("clean", "poisoned")
        ]
        errors = [0] + [
            result["summary"][variant][metric]["std"] for variant in ("clean", "poisoned")
        ]
        ax.bar(names, means, yerr=errors, capsize=5, color=["#64748b", "#0f766e", "#dc2626"])
        ax.set(title=title, ylim=(0, 1.1), ylabel="Доля")
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.2)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    png = buffer.getvalue()
    (output / "qwen-poisoning.png").write_bytes(png)
    rows = []
    for name, metrics in [("Исходная Qwen", result["base"]["metrics"])]:
        rows.append(
            f"<tr><td>{name}</td>"
            + "".join(
                f"<td>{metrics[m]:.1%}</td>"
                for m in ("accuracy", "macro_f1", "source_to_target_rate", "invalid_rate")
            )
            + "</tr>"
        )
    for name, variant in zip(names[1:], ("clean", "poisoned"), strict=True):
        metrics = result["summary"][variant]
        rows.append(
            f"<tr><td>{name}</td>"
            + "".join(
                f"<td>{metrics[m]['mean']:.1%} ± {metrics[m]['std']:.1%}</td>"
                for m in ("accuracy", "macro_f1", "source_to_target_rate", "invalid_rate")
            )
            + "</tr>"
        )
    runs = "".join(
        f"<tr><td>{row['seed']}</td><td>{row['variant']}</td>"
        f"<td>{row['metrics']['accuracy']:.1%}</td>"
        f"<td>{row['metrics']['source_to_target_rate']:.1%}</td></tr>"
        for row in result["runs"]
    )
    plan = result["plan"]
    page = f"""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Qwen: отравление дообучения</title>
<style>body{{background:#eef2f7;color:#172033;font:17px/1.65 system-ui;margin:0}}
main{{max-width:1100px;margin:35px auto;background:white;padding:32px;border-radius:18px}}
h1{{font-size:clamp(28px,5vw,44px);line-height:1.2}}img{{width:100%}}
table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;text-align:left;
border-bottom:1px solid #dbe2eb}}.scroll{{overflow:auto}}th{{background:#f1f5f9}}
pre{{padding:18px;background:#f1f5f9;overflow:auto}}.note{{border-left:4px solid #dc2626;
padding:10px 20px}}@media(max-width:650px){{main{{margin:0;padding:18px}}}}</style>
<main><p>ML POISONING LAB / QWEN</p><h1>Ложные обучающие ответы меняют маршрутизацию заявок?</h1>
<p>{html.escape(plan["model"])} · LoRA · {plan["repeats"]} парных повтора · локальный MLX</p>
<p>180 вымышленных обучающих заявок, 36 validation, 54 test. Атака меняет A → B у 54 строк:
30% всего train и 90% сетевых заявок. Test и validation остаются чистыми.</p>
<img alt="Сравнение исходной Qwen, чистого и отравленного LoRA"
src="data:image/png;base64,{base64.b64encode(png).decode()}">
<div class="scroll"><table><tr><th>Вариант</th><th>Accuracy</th><th>Macro F1</th>
<th>A → B</th><th>Invalid</th></tr>{"".join(rows)}</table></div>
<p>A — сеть; B — учётная запись; C — оборудование. A → B считается среди истинных A.
Invalid-ответы включены в ошибки. ± означает выборочный SD, а не доверительный интервал.
Исходная Qwen оценена один раз; адаптеры — в трёх повторах.</p>
<h2>Все повторы</h2><table><tr><th>Seed</th><th>Вариант</th><th>Accuracy</th><th>A → B</th></tr>
{runs}</table><h2>Границы выводов</h2><div class="note">
Одна небольшая модель, один синтетический датасет, высокий доступ атакующего к одному классу.
Test использует только два шаблона на класс. Подмена также меняет баланс обучающих ответов.
Это не скрытый backdoor и не доказательство уязвимости корпоративных установок Qwen.</div>
<h2>Воспроизводимость</h2><p>Все предсказания, метрики, seed и параметры находятся в results.json.
Веса и адаптеры хранятся локально, отдельно от Git.</p>
<pre>{html.escape(json.dumps(result["environment"], ensure_ascii=False, indent=2))}</pre>
</main></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Отчёт по выполненному эксперименту Qwen")
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_report(json.loads(args.results.read_text()), args.output)


if __name__ == "__main__":
    main()
