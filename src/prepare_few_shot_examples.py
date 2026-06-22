import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "few_shot" / "qwen_4shot_train_examples.jsonl"

SHOT_SPECS = [
    {
        "name": "incident_outage_high",
        "ticket_type": "Incident",
        "topic": "Service Outages and Maintenance",
        "urgency": "high",
    },
    {
        "name": "request_customer_service_low",
        "ticket_type": "Request",
        "topic": "Customer Service",
        "urgency": "low",
    },
    {
        "name": "problem_product_support_medium",
        "ticket_type": "Problem",
        "topic": "Product Support",
        "urgency": "medium",
    },
    {
        "name": "change_it_support_medium",
        "ticket_type": "Change",
        "topic": "IT Support",
        "urgency": "medium",
    },
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_shortest_example(rows: dict, spec: dict) -> dict:
    candidates = []

    for row in rows:
        target = row["target"]

        matched_spec = (
            target["ticket_type"] == spec["ticket_type"]
            and target["topic"] == spec["topic"]
            and target["urgency"] == spec["urgency"]
        )

        if matched_spec:
            candidates.append(row)

    if not candidates:
        raise ValueError(f"No train example found for: {spec}")

    return min(
        candidates,
        key=lambda row: len(row["messages"][1]["content"]),
    )


def main() -> None:
    train_rows = read_jsonl(TRAIN_PATH)
    few_shot_rows = []

    for spec in SHOT_SPECS:
        selected = select_shortest_example(train_rows, spec)

        few_shot_rows.append(
            {
                "selection_name": spec["name"],
                "source_split": "train",
                "target": selected["target"],
                "messages": [
                    selected["messages"][1],
                    selected["messages"][2],
                ]
            }
        )

    save_jsonl(few_shot_rows, OUTPUT_PATH)

    print(f"Train rows: {len(train_rows)}")
    print(f"Saved few-shot examples: {len(few_shot_rows)}")
    print(f"Output path: {OUTPUT_PATH}")

    for index, row in enumerate(few_shot_rows, start=1):
        print("\n" + "=" * 80)
        print(f"Shot {index}: {row['selection_name']}")
        print(row["messages"][0]["content"])
        print(json.dumps(row["target"], ensure_ascii=False))

if __name__ == "__main__":
    main()