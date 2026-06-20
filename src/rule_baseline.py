import json
from pathlib import Path

from src.prediction_utils import parse_and_validate_response


PROJECT_DIR = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def predict_one(text: str) -> dict:
    text_lower = text.lower()

    ticket_type = "Request"
    topic = "Technical Support"
    urgency = "medium"
    tags = []

    if any(word in text_lower for word in ["crash", "error", "bug", "issue", "problem", "failed", "failure"]):
        ticket_type = "Incident"

    if any(word in text_lower for word in ["password", "login", "account", "access"]):
        topic = "Technical Support"
        tags.append("Account")

    if any(word in text_lower for word in ["invoice", "payment", "billing", "charge", "refund"]):
        topic = "Billing and Payments"
        tags.append("Billing")

    if any(word in text_lower for word in ["refund", "return", "exchange"]):
        topic = "Returns and Exchanges"
        tags.append("Refund")

    if any(word in text_lower for word in ["outage", "downtime", "offline", "unavailable"]):
        topic = "Service Outages and Maintenance"
        ticket_type = "Incident"
        tags.append("Outage")

    if any(word in text_lower for word in ["urgent", "asap", "immediately", "critical", "blocked"]):
        urgency = "high"

    if any(word in text_lower for word in ["documentation", "guide", "manual", "api"]):
        tags.append("Documentation")

    if any(word in text_lower for word in ["software", "application", "system", "technical"]):
        tags.append("Tech Support")

    tags = sorted(set(tags))

    return {
        "ticket_type": ticket_type,
        "topic": topic,
        "urgency": urgency,
        "tags": tags,
    }


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def main() -> None:
    input_path = PROJECT_DIR / "data" / "processed" / "test.jsonl"
    output_path = PROJECT_DIR / "predictions" / "rule_baseline.jsonl"

    rows = read_jsonl(input_path)

    predictions = []

    for row in rows:
        text = row['messages'][1]['content']
        prediction = predict_one(text)

        raw_response = json.dumps(prediction, ensure_ascii=False)
        result = parse_and_validate_response(raw_response)

        predictions.append(result)

    save_jsonl(predictions, output_path)

    print(f"Input rows: {len(rows)}")
    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()