# Dataset Card: Structured Support Ticket Routing

## Описание задачи

Задача проекта — научить LLM извлекать структурированную информацию из клиентских обращений в поддержку.

На вход модель получает текст тикета:

- `subject`;
- `body`.

На выходе модель должна вернуть строгий JSON:

```json
{
  "ticket_type": "Request",
  "topic": "Technical Support",
  "urgency": "high",
  "tags": ["Product", "Feature", "Documentation", "Tech Support"]
}
```

## Источник данных

Используется датасет:

`Tobi-Bueck/customer-support-tickets`

Из исходного датасета используются поля:

- `subject` — тема обращения;
- `body` — текст обращения;
- `type` — тип тикета;
- `queue` — отдел/очередь для маршрутизации;
- `priority` — срочность;
- `tag_1` ... `tag_8` — теги обращения;
- `language` — язык обращения.

Поле `answer` не используется, потому что проект решает задачу structured extraction / ticket routing, а не генерацию ответа клиенту.

## Фильтрация данных

В текущей версии используются только англоязычные обращения:

```python
df_en = df[df["language"] == "en"]
```

Для быстрой разработки используется подвыборка:

```python
MAX_EXAMPLES = 1500
```

## Формат данных

Каждый пример сохраняется в chat-format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You extract structured routing information from customer support tickets. Return only valid JSON with fields: ticket_type, topic, urgency, tags."
    },
    {
      "role": "user",
      "content": "Subject: ...\n\nBody: ..."
    },
    {
      "role": "assistant",
      "content": "{\"ticket_type\":\"Request\",\"topic\":\"Technical Support\",\"urgency\":\"high\",\"tags\":[\"Product\"]}"
    }
  ],
  "target": {
    "ticket_type": "Request",
    "topic": "Technical Support",
    "urgency": "high",
    "tags": ["Product"]
  }
}
```

`messages` используется для supervised fine-tuning, а `target` — для валидации и подсчёта метрик.

## JSON schema

Выход модели должен соответствовать схеме:

```json
{
  "ticket_type": "Incident | Request | Problem | Change",
  "topic": "non-empty string",
  "urgency": "low | medium | high | critical",
  "tags": ["string"]
}
```

## Split

Текущая dev-версия датасета:

- train: 1200 examples;
- validation: 150 examples;
- test: 150 examples.

Split создаётся в пропорции:

- train: 80%;
- validation: 10%;
- test: 10%.

Используется `random_state=42`.

## Валидация

Датасет проверяется скриптом:

```bash
python -m src.validate_dataset
```

Проверяются:

- валидность каждой JSONL-строки;
- наличие полей `messages` и `target`;
- структура ролей `system`, `user`, `assistant`;
- соответствие `target` Pydantic-схеме;
- валидность JSON внутри `assistant.content`;
- соответствие `assistant.content` той же Pydantic-схеме.

Текущий результат валидации:

```text
data/processed/train.jsonl: total=1200, errors=0
data/processed/validation.jsonl: total=150, errors=0
data/processed/test.jsonl: total=150, errors=0
```

## Планируемые метрики

Для оценки моделей будут использоваться:

- JSON validity;
- schema validity;
- ticket_type accuracy / macro-F1;
- topic accuracy / macro-F1;
- urgency accuracy / macro-F1;
- tag precision / recall / F1;
- exact match.

## Ограничения

- Текущая версия использует только 1500 примеров для быстрой разработки.
- Используются только англоязычные обращения.
- Качество тегов зависит от исходных колонок `tag_1` ... `tag_8`.
- Проект не оценивает качество ответа клиенту, так как поле `answer` не используется.
- Основная цель датасета — обучение и оценка structured JSON extraction.
