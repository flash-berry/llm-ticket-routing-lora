# Error Analysis — Qwen 4-Shot Prompt Baseline

## Цель анализа

Этот документ разбирает ошибки Qwen2.5-0.5B-Instruct в **4-shot prompting** режиме для задачи structured ticket routing.

Модель получает:

1. строгую system instruction с допустимыми значениями;
2. четыре фиксированных demonstration-примера, выбранных только из `train.jsonl`;
3. новый тикет из `test.jsonl`.

Задача модели — вернуть JSON:

```json
{
  "ticket_type": "Incident | Request | Problem | Change",
  "topic": "...",
  "urgency": "low | medium | high",
  "tags": ["..."]
}
```

Анализ проведён на 150 примерах test split. Few-shot примеры не пересекаются с test split.

---

## Метрики

| Метрика | Qwen zero-shot | Qwen 4-shot |
|---|---:|---:|
| ticket_type_accuracy | 0.3200 | 0.5267 |
| ticket_type_macro_f1 | 0.1784 | 0.5154 |
| topic_accuracy | 0.0133 | 0.1133 |
| topic_macro_f1 | 0.0286 | 0.1888 |
| urgency_accuracy | 0.4533 | 0.3333 |
| urgency_macro_f1 | 0.2951 | 0.3337 |
| tag_precision | 0.0018 | 0.1633 |
| tag_recall | 0.0013 | 0.1497 |
| tag_f1 | 0.0015 | 0.1562 |
| exact_match | 0.0000 | 0.0000 |
| exact_match_without_tags | 0.0000 | 0.0267 |
| json_validity | 1.0000 | 1.0000 |
| schema_validity | 0.0067 | 0.9067 |

## Главный результат

Few-shot prompting заметно улучшил соблюдение формата и качество нескольких полей:

- `schema_validity` выросла с `0.0067` до `0.9067`: полностью валидны 136 из 150 ответов;
- `ticket_type_macro_f1` вырос с `0.1784` до `0.5154`;
- `topic_macro_f1` вырос с `0.0286` до `0.1888`;
- `tag_f1` вырос почти с нуля до `0.1562`.

При этом topic остаётся главным слабым местом: модель часто использует `General Inquiry` как универсальный ответ. Также few-shot не улучшил urgency accuracy относительно zero-shot.

---

## 1. Распределение предсказанных topic

| Predicted topic | Количество |
|---|---:|
| General Inquiry | 106 |
| IT Support | 15 |
| Technical Support | 9 |
| Billing and Payments | 6 |
| Customer Service | 3 |
| Returns and Exchanges | 2 |
| Human Resources | 2 |
| Marketing | 1 |
| Product Support | 1 |
| Digital Branding | 1 |
| Healthcare | 1 |
| Documentation | 1 |
| Pre-Sales Query | 1 |
| Sales and Pre-Sales | 1 |

`General Inquiry` выбрана в 106 из 150 случаев, то есть в **70.67%** ответов.

### Интерпретация

Few-shot examples помогли модели почти всегда возвращать допустимый JSON-объект, но не устранили коллапс в слишком общий topic. `General Inquiry` становится «безопасной» категорией, когда модель не уверена в более конкретной маршрутизации.

Это объясняет низкую `topic_accuracy = 0.1133`: допустимое значение по схеме не обязательно является правильным значением для конкретного тикета.

---

## 2. Причины schema-invalid ответов

Всего невалидных по Pydantic-схеме ответов: **14 из 150**.

| Причина | Количество |
|---|---:|
| Недопустимое значение `topic` | 5 |
| Недопустимое значение `ticket_type` | 4 |
| 9 тегов при лимите максимум 8 | 4 |
| 11 тегов при лимите максимум 8 | 1 |

Сумма причин равна 14, поэтому в этом прогоне каждый schema-invalid ответ содержал одну зафиксированную причину нарушения.

### Интерпретация

Основной прогресс few-shot заключается в том, что модель почти научилась соблюдать schema-level contract. Оставшиеся ошибки относятся не к синтаксису JSON, а к ограничениям output space:

- модель иногда придумывает свободный topic;
- модель иногда использует свободный ticket type вместо одного из четырёх разрешённых;
- модель иногда превышает лимит `tags <= 8`.

---

## 3. Анализ тегов

| Показатель | Значение |
|---|---:|
| Ответов с `tags` как list | 150 |
| Ответов с отсутствующим ключом `tags` | 0 |
| Ответов с `tags` не в формате list | 0 |
| Среднее число тегов | 4.63 |
| Ответов с более чем 8 тегами | 5 |

Распределение длины списка тегов:

| Количество тегов | Количество ответов |
|---|---:|
| 0 | 9 |
| 2 | 3 |
| 3 | 22 |
| 4 | 34 |
| 5 | 37 |
| 6 | 28 |
| 7 | 10 |
| 8 | 2 |
| 9 | 4 |
| 11 | 1 |

### Интерпретация

Модель стабильно возвращает `tags` в правильном типе данных: список присутствует во всех 150 ответах. Это существенное улучшение структуры по сравнению с zero-shot.

Однако качество тегов ограничено:

- `tag_precision = 0.1633`;
- `tag_recall = 0.1497`;
- `tag_f1 = 0.1562`.

Few-shot дал модели представление о стиле тегов, но не научил стабильно воспроизводить конкретный gold label set. Кроме того, в 5 ответах модель нарушила ограничение на число тегов.

---

## 4. Основные путаницы в ticket_type

Наиболее частые ошибки:

| Gold ticket type | Predicted ticket type | Количество |
|---|---|---:|
| Incident | Problem | 38 |
| Change | Request | 10 |
| Incident | Request | 9 |
| Problem | Request | 4 |
| Problem | Incident | 2 |
| Request | Change | 2 |

Также в нескольких ответах были свободные, недопустимые значения: `Issue Report`, `Issue`, `Issue with Automated Rebalancing Systems`, `Advice`.

### Интерпретация

Главная путаница — `Incident → Problem`. Для поддержки это близкие, но не идентичные классы:

- `Incident` обычно описывает конкретное нарушение работы здесь и сейчас;
- `Problem` обычно обозначает более широкую или повторяющуюся первопричину.

Маленькая instruction-модель без supervised training не всегда различает это разграничение по правилам разметки конкретного датасета.

---

## 5. Основные путаницы в topic

Наиболее частые ошибки:

| Gold topic | Predicted topic | Количество |
|---|---|---:|
| Technical Support | General Inquiry | 32 |
| Product Support | General Inquiry | 29 |
| Customer Service | General Inquiry | 14 |
| IT Support | General Inquiry | 8 |
| Service Outages and Maintenance | General Inquiry | 8 |
| Billing and Payments | General Inquiry | 5 |
| Product Support | IT Support | 4 |
| Sales and Pre-Sales | General Inquiry | 4 |
| Technical Support | IT Support | 4 |
| Customer Service | IT Support | 3 |

### Интерпретация

Модель чаще всего выбирает `General Inquiry` вместо конкретной функциональной очереди. Это подтверждает, что few-shot примеры помогли с форматом, но не дали достаточного представления о границах между 10 категориями topic.

Дополнительно модель иногда заменяет допустимые классы на семантически похожие, но запрещённые значения: `Marketing`, `Digital Branding`, `Healthcare`, `Documentation`, `Pre-Sales Query`.

---

## Почему exact match остаётся равным 0

`exact_match` требует совпадения всех полей:

- `ticket_type`;
- `topic`;
- `urgency`;
- полного набора tags.

Несмотря на рост валидности формата, модель одновременно редко угадывает все четыре части ответа. Более мягкая метрика `exact_match_without_tags = 0.0267` показывает, что только в 4 из 150 случаев модель одновременно верно определила `ticket_type`, `topic` и `urgency`.

---

## Вывод

Few-shot prompting оказался полезным промежуточным этапом:

1. Он почти устранил нарушения строгой JSON/Pydantic-схемы.
2. Он значительно улучшил balanced quality для `ticket_type`.
3. Он сделал генерацию тегов структурно стабильной и повысил tag F1.
4. Он не решил основную задачу маршрутизации по `topic`: Qwen слишком часто выбирает `General Inquiry`.
5. Он не устранил путаницу между `Incident` и `Problem`.

Следующий этап — QLoRA fine-tuning на `train.jsonl` с контролем по `validation.jsonl`. Обучение должно помочь модели освоить именно dataset-specific границы классов, а не только формат ответа.
