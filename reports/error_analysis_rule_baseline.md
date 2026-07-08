# Error Analysis v2 — Rule Baseline

## Цель анализа

Цель анализа — понять, где ошибается rule baseline, который предсказывает `ticket_type`, `topic`, `urgency` и `tags` по ключевым словам.

Rule baseline используется как детерминированная нижняя точка сравнения перед prompt-only LLM, few-shot prompting и LoRA/QLoRA fine-tuning.

## Данные и подготовка gold labels

Оценка выполнена на test split из 150 примеров.

Перед разбиением данных при подготовке датасета теги внутри каждого исходного примера были очищены от дубликатов: сохраняется первое появление тега и его исходный порядок. После этого обработанные `train`, `validation` и `test` файлы были проверены Pydantic-схемой.

Это важно для интерпретации метрик:

- gold tags в test split не содержат повторов;
- повтор одного и того же тега не может искусственно увеличить число совпадений или ошибок;
- правило `tags must not contain duplicates` одинаково применяется к gold labels и к model outputs.

Очистка относится к эталонным данным. Она не делает rule baseline сильнее и не изменяет его правила предсказания.

## Метрики rule baseline

Результаты на test split из 150 примеров:

| Метрика | Значение |
|---|---:|
| ticket_type_accuracy | 0.6467 |
| ticket_type_macro_f1 | 0.3834 |
| topic_accuracy | 0.3467 |
| topic_macro_f1 | 0.2263 |
| urgency_accuracy | 0.4867 |
| urgency_macro_f1 | 0.3185 |
| tag_precision | 0.5000 |
| tag_recall | 0.1060 |
| tag_f1 | 0.1749 |
| exact_match | 0.0000 |
| exact_match_without_tags | 0.1067 |
| json_validity | 1.0000 |
| schema_validity | 1.0000 |

Rule baseline создаёт JSON программно и использует допустимые значения из закрытой схемы. Поэтому все 150 ответов синтаксически валидны и проходят schema validation.

## Общая картина ошибок

Все 150 test-примеров имеют хотя бы одно несовпадение с gold target. Это не означает, что baseline полностью ошибается в каждом тикете: достаточно одного неверного поля или одного тега, чтобы пример попал в error analysis.

Количество ошибок по полям:

| Поле | Количество ошибок |
|---|---:|
| tags | 150 |
| topic | 98 |
| urgency | 77 |
| ticket_type | 53 |

Самое слабое поле — `tags`. При точном сравнении множества предсказанных и gold tags не совпали ни в одном примере.

## Основные типы ошибок

### 1. Ошибки в tags

Rule baseline ошибается в тегах во всех 150 примерах. Набор правил покрывает только небольшую часть возможных тегов и не извлекает бизнес-смысл текста.

Пример:

```text
Subject: Payment Details for Billing
```

Gold:

```json
{
  "tags": ["Billing", "Payment", "Invoice", "Hardware", "Software", "Procurement"]
}
```

Prediction:

```json
{
  "tags": ["Account", "Billing", "Documentation", "Tech Support"]
}
```

Baseline нашёл `Billing`, но добавил нерелевантные теги и пропустил `Payment`, `Invoice`, `Hardware`, `Software` и `Procurement`.

Высокая precision при низком recall означает: когда baseline выдаёт тег, он сравнительно часто релевантен, но baseline покрывает лишь малую часть всех нужных тегов.

### 2. Ошибки в topic

`topic` предсказан неверно в 98 из 150 примеров. Основная причина — fallback к `Technical Support` при отсутствии короткого списка знакомых ключевых слов.

Пример:

```text
Subject: Revise Digital Marketing Approaches
```

Gold:

```json
{
  "topic": "Product Support"
}
```

Prediction:

```json
{
  "topic": "Technical Support"
}
```

Baseline не умеет соотносить маркетинговую стратегию и обновление продукта с business taxonomy без явно прописанного правила.

Отдельно важно: некоторые gold labels сами выглядят неинтуитивно по тексту. Например, запрос об оптимизации SQL Server в test split размечен как `Billing and Payments`. Поэтому часть ошибок отражает не только ограниченность правил, но и сложность исходной разметки.

### 3. Ошибки в urgency

`urgency` предсказан неверно в 77 из 150 примеров. Правила реагируют преимущественно на явные маркеры: `urgent`, `asap`, `critical`, `blocked`.

Пример:

```text
Subject: Service Interruption
```

Gold:

```json
{
  "urgency": "high"
}
```

Prediction:

```json
{
  "urgency": "medium"
}
```

Описание содержит признаки высокой срочности — сервис недоступен и есть операционные нарушения, — но правило не видит многие семантические формулировки срочности.

### 4. Ошибки в ticket_type

`ticket_type` предсказан неверно в 53 из 150 примеров. Baseline часто путает `Request`, `Change`, `Problem` и `Incident`.

Пример:

```text
Subject: Revise Digital Marketing Approaches
```

Gold:

```json
{
  "ticket_type": "Change"
}
```

Prediction:

```json
{
  "ticket_type": "Request"
}
```

Запрос на изменение стратегии действительно сформулирован как просьба, поэтому простое правило выбирает `Request`, хотя gold label относит его к `Change`.

Другой пример:

```text
Subject: Service Interruption
```

Gold:

```json
{
  "ticket_type": "Problem"
}
```

Prediction:

```json
{
  "ticket_type": "Incident"
}
```

Текст похож на инцидент, но в разметке он относится к `Problem`. Это показывает, что границы между классами не всегда выводятся из одиночных ключевых слов.

## Почему exact_match равен 0

`exact_match` требует полного совпадения:

- `ticket_type`;
- `topic`;
- `urgency`;
- множества `tags`.

Даже один неправильный или пропущенный тег делает пример неправильным по этой метрике. Поэтому `exact_match = 0.0` ожидаем при текущем качестве извлечения тегов.

Более мягкая метрика `exact_match_without_tags = 0.1067` показывает, что основные поля без тегов полностью совпали примерно в 10.7% test-примеров.

## Вывод

Rule baseline полезен как строгий и полностью валидный инженерный baseline, но его качество ограничено поиском ключевых слов.

Главные слабые стороны:

1. Очень низкий recall по тегам.
2. Частый fallback к `Technical Support`.
3. Слабое распознавание urgency без явных слов-маркеров.
4. Путаница между `Request`, `Change`, `Problem` и `Incident`.

Следующие сравнения должны проводиться на том же очищенном test split и через тот же evaluator. Это позволяет честно сопоставить rule baseline, Qwen zero-shot, few-shot prompting и LoRA/QLoRA.
