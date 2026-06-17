import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


MAX_EXAMPLES = 1500 # None - сохранить все примеры из датасета для train/val/test
SYSTEM_PROMPT = (
    "You extract structured routing information from customer support tickets. "
    "Return only valid JSON with fields: ticket_type, topic, urgency, tags."
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / 'processed'


ds = load_dataset("Tobi-Bueck/customer-support-tickets")

df = ds['train'].to_pandas()

df_en = df[df['language'] == 'en'].copy()


def build_input(row: pd.Series) -> str:
    for col in ['subject', 'body']:
        if col not in row.index:
            raise ValueError(f"Отсутсвует колонка {col}")

    subject = str(row.get('subject', "")).strip()
    body = str(row.get('body', "")).replace("\\n", "\n").strip()
    return f"Subject: {subject}\n\nBody: {body}"


def build_tags(row: pd.Series) -> list[str]:
    tag_cols = [f"tag_{i}" for i in range(1, 9)]

    tags = []

    for col in tag_cols:
        if col not in row.index:
            continue

        value = row.get(col)

        if pd.isna(value):
            continue

        value = str(value).strip()

        if not value:
            continue

        tags.append(value)

    return tags


def build_target(row: pd.Series) -> dict:
    for col in ['type', 'queue', 'priority']:
        if col not in row.index:
            raise ValueError(f"Отсутствует колонка {col}")

    return {
        "ticket_type": str(row.get('type', "")).strip(),
        "topic": str(row.get('queue', "")).strip(),
        "urgency": str(row.get('priority', "")).strip(),
        "tags": build_tags(row)
    }


def build_chat_example(row: pd.Series) -> dict:
    input_text = build_input(row)
    target = build_target(row)

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": input_text,
            },
            {
                "role": "assistant",
                "content": json.dumps(target, ensure_ascii=False)
            }
        ],
        "target": target,
    }


examples = [build_chat_example(row) for _, row in df_en.iterrows()]

if MAX_EXAMPLES is not None:
    examples = examples[:MAX_EXAMPLES]


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


train_rows, temp_rows = train_test_split(
    examples,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)

validation_rows, test_rows = train_test_split(
    temp_rows,
    test_size=0.5,
    random_state=42,
    shuffle=True,
)

save_jsonl(train_rows, OUTPUT_DIR / "train.jsonl")
save_jsonl(validation_rows, OUTPUT_DIR / "validation.jsonl")
save_jsonl(test_rows, OUTPUT_DIR / "test.jsonl")

print("=" * 80)
print("DATASET PREPARATION SUMMARY")
print("=" * 80)

print(f"Original rows: {len(df)}")
print(f"English rows: {len(df_en)}")
print(f"MAX_EXAMPLES: {MAX_EXAMPLES}")
print(f"Final examples: {len(examples)}")

print("\nColumns:")
print(df_en.columns.tolist())

print("\nTicket type distribution:")
print(df_en["type"].value_counts())

print("\nPriority distribution:")
print(df_en["priority"].value_counts())

print("\nQueue distribution:")
print(df_en["queue"].value_counts().head(20))

print("\nTag columns:")
print([col for col in df_en.columns if col.startswith("tag_")])

non_empty_tags = sum(1 for ex in examples if len(ex["target"]["tags"]) > 0)
avg_tags = sum(len(ex["target"]["tags"]) for ex in examples) / len(examples)

print("\nTags summary:")
print(f"Examples with non-empty tags: {non_empty_tags}/{len(examples)}")
print(f"Average tags per example: {avg_tags:.2f}")

print("\nExample with tags:")
for ex in examples:
    if ex["target"]["tags"]:
        print(json.dumps(ex["target"], indent=2, ensure_ascii=False))
        break

print("\nSplit sizes:")
print(f"train: {len(train_rows)}")
print(f"validation: {len(validation_rows)}")
print(f"test: {len(test_rows)}")

print("\nExample target:")
print(json.dumps(train_rows[0]["target"], indent=2, ensure_ascii=False))

print("\nOutput files:")
for path in [
    OUTPUT_DIR / "train.jsonl",
    OUTPUT_DIR / "validation.jsonl",
    OUTPUT_DIR / "test.jsonl",
]:
    print(f"{path}: {path.stat().st_size / 1024:.1f} KB")

print("=" * 80)