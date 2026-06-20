# Error Analysis v1 — Qwen Zero-Shot Baseline

## Цель анализа

Цель анализа — понять, как `Qwen/Qwen2.5-0.5B-Instruct` ведёт себя в zero-shot режиме на задаче маршрутизации support tickets.

Модель получает строгий system prompt с допустимыми значениями для:

- `ticket_type`;
- `topic`;
- `urgency`;
- `tags`.

Модель не дообучалась на train split. Она должна вернуть один JSON-объект для каждого тикета.

## Настройка эксперимента

- модель: `Qwen/Qwen2.5-0.5B-Instruct`;
- режим: zero-shot;
- генерация: greedy decoding (`temperature = 0.0`, `do_sample = False`);
- test split: 150 примеров;
- оценка: единый evaluator, используемый также для rule baseline.

Evaluator разделяет две группы метрик:

1. **Content quality** — насколько верно предсказаны отдельные поля из `parsed_response`.
2. **Format quality** — является ли ответ валидным JSON и проходит ли весь объект Pydantic-схему.

Это важно: частично правильный `ticket_type` получает зачёт в `ticket_type_accuracy`, даже когда весь ответ не проходит schema validation из-за неверного `topic` или тегов.

## Метрики Qwen zero-shot

| Метрика | Значение |
|---|---:|
| ticket_type_accuracy | 0.3200 |
| ticket_type_macro_f1 | 0.1784 |
| topic_accuracy | 0.0133 |
| topic_macro_f1 | 0.0286 |
| urgency_accuracy | 0.4533 |
| urgency_macro_f1 | 0.2951 |
| tag_precision | 0.0018 |
| tag_recall | 0.0013 |
| tag_f1 | 0.0015 |
| exact_match | 0.0000 |
| exact_match_without_tags | 0.0000 |
| json_validity | 1.0000 |
| schema_validity | 0.0067 |

`schema_validity = 0.0067` означает, что полностью валидным оказался 1 из 150 ответов.

## Главный результат

Qwen возвращает синтаксически корректный JSON во всех 150 случаях, но почти не соблюдает закрытую taxonomy задачи.

Это не означает, что модель всегда не понимает содержание тикета. Напротив, она часто формирует осмысленные свободные описания. Проблема в том, что эти описания не отображаются в заранее заданные классы `topic`, а tags часто не совпадают с каноническими gold labels.

## 1. Нарушение закрытой taxonomy для topic

В schema разрешены только 10 значений `topic`.

Результаты анализа:

| Показатель | Значение |
|---|---:|
| Допустимый topic | 2 / 150 |
| Недопустимый topic | 148 / 150 |

Наиболее частые недопустимые темы:

| Количество | Topic, сгенерированный Qwen |
|---:|---|
| 3 | Digital Marketing Strategies |
| 2 | Technical Issue |
| 2 | Digital Strategies |
| 2 | Integration Assistance |
| 2 | Service Disruption |

Модель чаще создаёт собственное семантическое название темы, чем выбирает один из допустимых классов.

Пример:

```text
Subject: Payment Details for Billing
```

Gold:

```json
{
  "ticket_type": "Request",
  "topic": "Billing and Payments",
  "urgency": "high"
}
```

Qwen:

```json
{
  "ticket_type": "Request",
  "topic": "Payment Details",
  "urgency": "high"
}
```

`Payment Details` семантически близко к billing, но не является точным допустимым значением `Billing and Payments`. Поэтому topic не засчитывается в строгой метрике и весь объект не проходит schema validation.

## 2. Сильный перекос в ticket_type

Распределение сгенерированных `ticket_type`:

| Ticket type | Количество |
|---|---:|
| Request | 132 |
| Problem | 18 |
| Incident | 0 |
| Change | 0 |

Модель почти всегда выбирает `Request`, даже для ситуаций, размеченных как `Incident` или `Change`.

Это объясняет низкий `ticket_type_macro_f1`: accuracy считает общее число попаданий, а macro-F1 дополнительно показывает слабое покрытие редких и непредсказанных классов.

## 3. Сильный перекос в urgency

Распределение сгенерированных `urgency`:

| Urgency | Количество |
|---|---:|
| high | 133 |
| medium | 16 |
| low | 1 |

Модель слишком часто выбирает `high`. Поэтому `urgency_accuracy = 0.4533` выглядит умеренно, но `urgency_macro_f1 = 0.2951` показывает плохой баланс между классами.

## 4. Теги: свободные формулировки вместо канонических labels

Tag F1 практически равен нулю:

```text
tag_f1 = 0.0015
```

Qwen обычно создаёт осмысленные, но свободные теги с другой формой, регистром или уровнем детализации.

Пример:

```text
Subject: Payment Details for Billing
```

Gold tags:

```json
["Billing", "Payment", "Invoice", "Hardware", "Software", "Procurement"]
```

Qwen tags:

```json
[
  "billing",
  "payments",
  "hardware",
  "software",
  "peripheral devices",
  "computers",
  "acquisitions",
  "accounting",
  "compliance",
  "internal procedures"
]
```

Смысл части тегов близок к gold tags, но текущая метрика сравнивает точные строки. Например, `billing` и `Billing` считаются разными тегами.

### Нарушения формата тегов

| Проверка | Количество ответов |
|---|---:|
| Более 8 тегов | 3 |
| Дублирующиеся теги | 0 |
| Пустые теги | 0 |
| Отсутствует поле `tags` | 3 |

Основная причина schema-invalid ответов — не tags, а недопустимый `topic`. Однако слишком длинные списки и пропущенный `tags` дополнительно нарушают контракт.

## 5. Почему JSON validity высокая, а schema validity низкая

Эти метрики отвечают на разные вопросы:

```text
json_validity = 1.0
```

означает, что модель во всех случаях сформировала текст, который успешно разбирается через `json.loads()`.

```text
schema_validity = 0.0067
```

означает, что почти все ответы нарушили хотя бы одно бизнес-правило схемы: обычно закрытый список `topic`, реже количество тегов или наличие обязательного поля `tags`.

Таким образом, Qwen умеет соблюдать JSON-синтаксис, но не умеет стабильно соблюдать прикладной контракт задачи.

## Сравнение с rule baseline

| Метрика | Rule baseline | Qwen zero-shot |
|---|---:|---:|
| Ticket type accuracy | 0.6467 | 0.3200 |
| Topic accuracy | 0.3467 | 0.0133 |
| Urgency accuracy | 0.4867 | 0.4533 |
| Tag F1 | 0.1749 | 0.0015 |
| JSON validity | 1.0000 | 1.0000 |
| Schema validity | 1.0000 | 0.0067 |
| Exact match without tags | 0.1067 | 0.0000 |

Rule baseline лучше на текущей задаче, потому что он по конструкции возвращает только значения из закрытого набора. Qwen zero-shot генерирует более естественные свободные формулировки, но они не совместимы с требованиями routing schema.

## Вывод

Zero-shot Qwen-0.5B не является пригодным решением для этой задачи без дополнительного управления выходом или обучения.

Основные проблемы:

1. Модель почти всегда придумывает `topic` вместо выбора из 10 разрешённых значений.
2. Модель сильно смещена в сторону `Request` и `high`.
3. Сгенерированные теги семантически свободны и почти не совпадают с каноническими gold labels.
4. Только 1 из 150 ответов полностью проходит schema validation.

Следующий эксперимент — few-shot prompting с примерами из train split. Он проверит, можно ли улучшить выбор закрытых классов без fine-tuning. После этого следует LoRA/QLoRA с тем же system prompt, тем же test split и тем же evaluator.
