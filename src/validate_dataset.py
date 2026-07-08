import json
from pathlib import Path

from pydantic import ValidationError

from src.schemas import TicketRoutingOutput


paths = [
    Path("data/processed/train.jsonl"),
    Path("data/processed/validation.jsonl"),
    Path("data/processed/test.jsonl"),
]


def validate_row(row: dict, line_number: int, path: Path) -> list[str]:
    errors = []

    if "messages" not in row:
        errors.append("missing field: messages")
        return errors

    if "target" not in row:
        errors.append("missing field: target")
        return errors

    messages = row["messages"]

    if not isinstance(messages, list):
        errors.append("messages is not a list")
        return errors

    if len(messages) != 3:
        errors.append(f"messages length is {len(messages)}, expected 3")
        return errors

    if not all(isinstance(message, dict) for message in messages):
        errors.append("each message must be a dict")
        return errors

    roles = [message.get("role") for message in messages]
    expected_roles = ["system", "user", "assistant"]

    if roles != expected_roles:
        errors.append(f"wrong roles: {roles}, expected {expected_roles}")

    target = row["target"]

    try:
        TicketRoutingOutput.model_validate(target)
    except ValidationError as exc:
        errors.append(f"target schema error: {exc.errors()}")

    assistant_content = messages[2].get("content")

    if not isinstance(assistant_content, str):
        errors.append("assistant_content is not a string")
        return errors

    try:
        assistant_json = json.loads(assistant_content)
    except json.JSONDecodeError as exc:
        errors.append(f"assistant content is not valid JSON: {exc}")
        return errors

    try:
        TicketRoutingOutput.model_validate(assistant_json)
    except ValidationError as exc:
        errors.append(f"assistant content schema error: {exc.errors()}")

    return errors


def validate_file(path: Path) -> None:
    total = 0
    error_count = 0
    shown_errors = 0
    max_shown_errors = 5

    if not path.exists():
        print(f"{path}: does not exist")
        return

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            total += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                error_count += 1

                if shown_errors < max_shown_errors:
                    print(f"\n{path}, line {line_number}")
                    print(f"JSON decode error: {exc}")
                    shown_errors += 1

                continue

            errors = validate_row(row, line_number, path)

            if errors:
                error_count += 1

                if shown_errors < max_shown_errors:
                    print(f"\n{path}, line {line_number}")
                    for error in errors:
                        print(f"- {error}")
                    shown_errors += 1

    print(f"{path}: total={total}, errors={error_count}")


def main() -> None:
    print("=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)

    for path in paths:
        validate_file(path)

    print("=" * 80)


if __name__ == "__main__":
    main()