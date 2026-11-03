"""Самодостаточный HTML и график без сетевых ресурсов."""

import base64
import html
import io
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def write_report(result, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    figure = Figure(figsize=(10, 5.4), layout="constrained", facecolor="#f8fafc")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    for name, label, color in (
        ("accuracy", "Accuracy", "#2563eb"),
        ("f1", "F1 (class 1)", "#0f766e"),
    ):
        ax.errorbar(
            [row["fraction"] * 100 for row in summary],
            [row[name]["mean"] for row in summary],
            yerr=[row[name]["std"] for row in summary],
            label=label,
            marker="o",
            capsize=4,
            color=color,
        )
    ax.set(
        xlabel="Подменённые обучающие метки, %",
        ylabel="Качество на чистом тесте",
        title="Случайная подмена меток: среднее и стандартное отклонение",
        ylim=(-0.05, 1.05),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    png = buffer.getvalue()
    (output / "poisoning.png").write_bytes(png)
    encoded = base64.b64encode(png).decode("ascii")
    rows = "".join(
        f"<tr><td>{row['fraction']:.0%}</td>"
        f"<td>{row['accuracy']['mean']:.3f} ± {row['accuracy']['std']:.3f}</td>"
        f"<td>{row['f1']['mean']:.3f} ± {row['f1']['std']:.3f}</td>"
        f"<td>{row['accuracy']['paired_delta_mean'] * 100:+.2f}</td></tr>"
        for row in summary
    )
    config = result["config"]
    environment = html.escape(json.dumps(result["environment"], ensure_ascii=False, indent=2))
    page = f"""<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ML Poisoning Lab — результаты эксперимента</title>
<style>
body{{margin:0;background:#eef2f7;color:#172033;font:17px/1.65 system-ui,sans-serif}}
main{{max-width:1060px;margin:40px auto;padding:36px;background:white;border-radius:20px}}
h1{{font-size:clamp(30px,5vw,48px);line-height:1.15;margin:12px 0}}h2{{font-size:24px}}
.tag{{color:#2563eb;font-weight:700;letter-spacing:.08em;font-size:13px}}
.cards{{display:flex;gap:16px;flex-wrap:wrap;margin:28px 0}}
.card{{background:#f1f5f9;padding:16px 22px;border-radius:12px;flex:1;min-width:150px}}
.card b{{display:block;font-size:26px}}img{{width:100%;height:auto}}
table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:12px;border-bottom:1px solid #dbe2eb}}
th{{background:#f1f5f9}}.table{{overflow-x:auto}}
.note{{border-left:4px solid #2563eb;padding:8px 20px}}
pre{{overflow-x:auto;background:#f1f5f9;padding:16px;border-radius:10px}}
@media(max-width:650px){{main{{margin:0;padding:20px;border-radius:0}}th,td{{padding:8px}}}}
</style><main>
<span class="tag">ML POISONING LAB / 0.1</span>
<h1>Что происходит, если испортить ответы в обучающих данных?</h1>
<p>Случайная подмена меток у синтетического бинарного классификатора.
Атакующий меняет обучающие ответы, а качество проверяется на чистых примерах.</p>
<div class="cards"><div class="card"><b>{config["samples"]}</b>синтетических примеров</div>
<div class="card"><b>{config["repeats"]}</b>парных повторов</div>
<div class="card"><b>70 / 30</b>обучение / тест</div></div>
<img alt="Качество модели при увеличении доли подменённых меток"
src="data:image/png;base64,{encoded}">
<h2>Измерения</h2><div class="table"><table>
<thead><tr><th>Подмена</th><th>Accuracy ± SD</th><th>F1 ± SD</th>
<th>Δ accuracy, п.п.</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p>Accuracy — доля правильных ответов. F1 объединяет точность и полноту для класса 1.
SD — стандартное отклонение между повторами, а не доверительный интервал.
Δ — среднее изменение относительно чистой модели того же повтора.</p>
<h2>Почему сравнение устроено так</h2>
<p>Внутри повтора данные, разделение на обучение и тест, модель и порядок выбора подмен
фиксированы. Больший бюджет продолжает тот же список выбранных индексов.
StandardScaler обучается только на обучающей выборке через Pipeline.
Разные повторы меняют генерацию данных, разделение и выбор подмен одновременно.</p>
<div class="note"><b>Границы вывода.</b> Это один синтетический сценарий и одна модель.
Случайная подмена не равна оптимизированной атаке. Падение качества не обязано быть
монотонным. Полная инверсия меток — контрольная крайность, а не реалистичный бюджет.
Защита от poisoning в этой версии не реализована.</div>
<h2>Воспроизводимость</h2><p>Параметры, seed, хеши выборок, матрицы ошибок и все отдельные
измерения записаны в соседний results.json.
HTML содержит график внутри себя и открывается офлайн.</p>
<pre>{environment}</pre></main></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
