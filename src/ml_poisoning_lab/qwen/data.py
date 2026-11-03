"""Вымышленные заявки: разные шаблоны для train, validation и test."""

import copy
import hashlib
import json
import math
import random
from pathlib import Path

LABELS = ("A", "B", "C")
SYSTEM = (
    "Определи категорию заявки в техподдержку. "
    "A — сеть и подключение к интернету. B — учётная запись и доступ. "
    "C — неисправность оборудования. Ответь только одной буквой: A, B или C."
)

# 0:6 — train, 6:8 — validation, 8:10 — ранняя диагностика, 10:12 — финальный test.
TEMPLATES = {
    "A": [
        "В {place} пропало подключение к интернету.",
        "В {place} компьютер не получает IP-адрес от DHCP.",
        "В {place} Wi-Fi постоянно теряет соединение.",
        "В {place} не разрешаются имена сайтов через DNS.",
        "В {place} очень большие задержки при передаче по сети.",
        "В {place} сетевой шлюз не отвечает на ping.",
        "В {place} не удаётся установить соединение с VPN.",
        "В {place} отсутствует связь между компьютерами локальной сети.",
        "В {place} пакеты до маршрутизатора теряются, кабель проверен.",
        "В {place} браузер сообщает о недоступности всех сайтов, сеть без доступа.",
        "В {place} компьютеры не видят друг друга после изменения маршрутов в сети.",
        "В {place} соединение с интернетом обрывается каждые несколько минут на всех устройствах.",
    ],
    "B": [
        "В {place} пользователь забыл пароль от учётной записи.",
        "В {place} учётная запись заблокирована после попыток входа.",
        "В {place} требуется выдать пользователю доступ к папке.",
        "В {place} нужно сбросить пароль для входа в систему.",
        "В {place} сотрудник не может пройти двухфакторную аутентификацию.",
        "В {place} требуется создать учётную запись сотрудника.",
        "В {place} срок действия пароля истёк.",
        "В {place} пользователю отказано в доступе из-за недостаточных прав.",
        "В {place} уволенному сотруднику нужно отключить профиль входа.",
        "В {place} после смены фамилии требуется переименовать логин сотрудника.",
        "В {place} сотруднику нужно восстановить доступ к своему профилю после блокировки.",
        "В {place} новому сотруднику требуется назначить права на общий ресурс.",
    ],
    "C": [
        "В {place} монитор не включается, индикатор питания не горит.",
        "В {place} клавиатура перестала реагировать на нажатия.",
        "В {place} принтер зажевал бумагу и сломал ролик подачи.",
        "В {place} жёсткий диск издаёт щелчки и не определяется.",
        "В {place} у ноутбука сломан разъём зарядки.",
        "В {place} мышь физически повреждена и требует замены.",
        "В {place} системный блок перегревается из-за сломанного вентилятора.",
        "В {place} у монитора разбита матрица.",
        "В {place} аккумулятор ноутбука вздулся и корпус деформирован.",
        "В {place} компьютер не запускается, блок питания вышел из строя.",
        "В {place} на экране появились полосы из-за повреждения дисплея.",
        "В {place} у принтера сломана крышка и повреждён механизм подачи листов.",
    ],
}
PREFIXES = ("Добрый день. ", "Помогите, пожалуйста: ", "Новая заявка. ", "Проблема: ", "")
SUFFIXES = (" Проверьте, пожалуйста.", " Нужна помощь специалиста.", " Просьба разобраться.")


def canonical_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def build_dataset(seed=42):
    """Разделение по шаблонам: одинаковое описание не переходит из train в test."""
    rng = random.Random(seed)
    result = {}
    for split, start, stop, per_template in (
        ("train", 0, 6, 10),
        ("valid", 6, 8, 6),
        ("test", 10, 12, 9),
    ):
        rows = []
        for label in LABELS:
            for template_index in range(start, stop):
                for variation in range(per_template):
                    text = TEMPLATES[label][template_index].format(
                        place=f"тестовом кабинете {100 + variation}",
                    )
                    text = rng.choice(PREFIXES) + text + rng.choice(SUFFIXES)
                    rows.append(
                        {
                            "id": f"{split}-{label}-{template_index}-{variation}",
                            "template_id": f"{label}-{template_index}",
                            "text": text,
                            "label": label,
                        }
                    )
        rng.shuffle(rows)
        result[split] = rows
    return result


def poison_training(rows, fraction, seed):
    """Подменить A→B в floor(n_train*fraction) строках; бюджет от всего train."""
    if not isinstance(fraction, (float, int)) or isinstance(fraction, bool):
        raise ValueError("fraction должен быть числом")
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction должен быть конечным числом от 0 до 1")
    eligible = [i for i, row in enumerate(rows) if row["label"] == "A"]
    count = math.floor(len(rows) * fraction)
    if count > len(eligible):
        raise ValueError("Бюджет превышает число сетевых заявок в train")
    random.Random(seed).shuffle(eligible)
    selected = eligible[:count]
    poisoned = copy.deepcopy(rows)
    for index in selected:
        poisoned[index]["label"] = "B"
    return poisoned, selected


def messages(row, include_answer=False):
    result = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["text"]}]
    if include_answer:
        result.append({"role": "assistant", "content": row["label"]})
    return result


def write_training_data(dataset, train_rows, tokenizer, output: Path):
    """Формат нашего runner, не вход для стандартного CLI mlx_lm.lora."""
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train_rows), ("valid", dataset["valid"])):
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                prompt = tokenizer.apply_chat_template(
                    messages(row),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                stream.write(
                    json.dumps({"prompt": prompt, "completion": row["label"]}, ensure_ascii=False)
                    + "\n"
                )
    # Test не передаётся процессу обучения даже как локальный файл.
