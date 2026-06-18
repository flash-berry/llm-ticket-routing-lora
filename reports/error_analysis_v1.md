# Error Analysis v1 — Rule Baseline

## Что означает v1

`v1` означает **version 1**, то есть первая версия анализа ошибок.

В этом файле разбирается только первый baseline — простой rule-based подход без LLM и без fine-tuning. Позже можно будет сделать:

- `error_analysis_v2.md` — для prompt-only LLM baseline;
- `error_analysis_v3.md` — для LoRA/QLoRA модели;
- `error_analysis_final.md` — для финального сравнения моделей.

## Цель анализа

Цель этого анализа — понять, где ошибается простой rule baseline, который предсказывает `ticket_type`, `topic`, `urgency` и `tags` по ключевым словам.

Rule baseline нужен как нижняя точка сравнения перед использованием LLM и LoRA/QLoRA fine-tuning.

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
| schema_validity | 1.0000 |

## Общая картина ошибок

В test split все 150 примеров имеют хотя бы одно несовпадение с эталонным ответом.

Количество ошибок по полям:

| Поле | Количество ошибок |
|---|---:|
| tags | 150 |
| topic | 98 |
| urgency | 77 |
| ticket_type | 53 |

Это показывает, что самая слабая часть rule baseline — извлечение тегов. Также baseline часто ошибается в маршрутизации тикета (`topic`) и определении срочности (`urgency`).

## Основные типы ошибок

### 1. Ошибки в tags

Baseline ошибается в тегах во всех 150 примерах. Причина в том, что набор правил покрывает только небольшую часть возможных тегов.

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

Baseline нашёл часть смысла через `Billing`, но добавил лишние теги и пропустил важные: `Payment`, `Invoice`, `Hardware`, `Software`, `Procurement`.

Вывод: для тегов простых ключевых слов недостаточно. Нужна модель, которая понимает контекст обращения.

## 2. Ошибки в topic

Topic ошибочно предсказан в 98 из 150 примеров. Основная проблема — baseline слишком часто выбирает дефолтный `Technical Support`.

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

Текст связан с обновлением маркетинговой стратегии и развитием продукта, но rule baseline не видит такие смысловые связи, если нет заранее заданных ключевых слов.

Вывод: topic требует понимания бизнес-контекста, а не только поиска слов вроде `login`, `payment`, `refund`.

## 3. Ошибки в urgency

Urgency ошибочно предсказан в 77 из 150 примеров. Baseline хорошо реагирует только на явные слова вроде `urgent`, `asap`, `critical`, `blocked`.

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

В тексте есть признаки высокой срочности: сервис недоступен, есть operational disruptions, пользователь просит immediate attention. Но baseline не всегда распознаёт такие формулировки.

Вывод: определение срочности требует анализа ситуации, а не только отдельных ключевых слов.

## 4. Ошибки в ticket_type

Ticket type ошибочно предсказан в 53 из 150 примеров. Чаще всего baseline путает `Request`, `Problem`, `Incident` и `Change`.

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

Запрос на изменение стратегии ближе к `Change`, но baseline по умолчанию часто выбирает `Request`.

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

Текст действительно похож на инцидент, но в разметке он относится к `Problem`. Это показывает, что различие между классами не всегда очевидно даже по тексту.

## Почему exact_match равен 0

`exact_match` требует полного совпадения всех полей:

- `ticket_type`;
- `topic`;
- `urgency`;
- `tags`.

Даже если baseline правильно угадал `ticket_type`, но ошибся в одном теге, весь пример считается неправильным.

Поэтому `exact_match = 0.0` ожидаем для rule baseline.

Более мягкая метрика `exact_match_without_tags = 0.1067` показывает, что без учёта тегов baseline полностью угадывает основные поля примерно в 10.7% случаев.

## Вывод

Rule baseline полезен как нижняя точка сравнения, но его качество ограничено.

Главные слабые места:

1. Низкий recall по тегам.
2. Слишком частое предсказание `Technical Support` как topic.
3. Слабое определение urgency без явных ключевых слов.
4. Путаница между `Request`, `Change`, `Problem` и `Incident`.

Следующий этап — prompt-only LLM baseline. Он должен показать, насколько open-source LLM без fine-tuning лучше понимает контекст обращения и умеет возвращать структурированный JSON.
