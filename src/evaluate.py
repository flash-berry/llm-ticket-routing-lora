import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from pydantic import ValidationError

from src.schemas import TicketRoutingOutput


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GOLD_PATH = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
DEFAULT_PRED_PATH = PROJECT_ROOT / "predictions" / "rule_baseline.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "rule_baseline_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ticket routing predictions."
    )

    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="Path to gold test JSONL file.",
    )

    parser.add_argument(
        "--pred-path",
        type=Path,
        default=DEFAULT_PRED_PATH,
        help="Path to prediction JSONL file.",
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to report JSONL file.",
    )

    return parser.parse_args()


def resolve_path(path:Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open('r', encoding='utf8') as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def get_targets_and_predictions(gold_rows: list[dict], pred_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    gold_targets = []
    predictions = []

    for gold_row, pred_row in zip(gold_rows, pred_rows):
        gold_targets.append(gold_row["target"])
        predictions.append(pred_row["prediction"])

    return gold_targets, predictions


def compute_metrics(
        gold_targets: list[dict],
        predictions: list[dict],
        field: str
) -> dict:
    y_true = [row[field] for row in gold_targets]
    y_pred = [row[field] for row in predictions]

    return {
        f"{field}_accuracy": accuracy_score(y_true, y_pred),
        f"{field}_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def compute_tag_metrics(
        gold_targets: list[dict],
        predictions: list[dict],
) -> dict:
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for gold, pred in zip(gold_targets, predictions):
        gold_tags = set(gold["tags"])
        pred_tags = set(pred["tags"])

        total_tp += len(gold_tags & pred_tags)
        total_fp += len(pred_tags - gold_tags)
        total_fn += len(gold_tags - pred_tags)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tag_precision": precision,
        "tag_recall": recall,
        "tag_f1": f1,
    }


def compute_exact_match(
    gold_targets: list[dict],
    predictions: list[dict],
) -> dict:
    matches = 0

    for gold, pred in zip(gold_targets, predictions):
        gold_for_compare = {
            "ticket_type": gold["ticket_type"],
            "topic": gold["topic"],
            "urgency": gold["urgency"],
            "tags": sorted(gold["tags"]),
        }

        pred_for_compare = {
            "ticket_type": pred["ticket_type"],
            "topic": pred["topic"],
            "urgency": pred["urgency"],
            "tags": sorted(pred["tags"]),
        }

        if gold_for_compare == pred_for_compare:
            matches += 1

    return {
        "exact_match": matches / len(gold_targets)
    }


def compute_exact_match_without_tags(
    gold_targets: list[dict],
    predictions: list[dict],
) -> dict:
    matches = 0

    for gold, pred in zip(gold_targets, predictions):
        gold_for_compare = {
            "ticket_type": gold["ticket_type"],
            "topic": gold["topic"],
            "urgency": gold["urgency"],
        }

        pred_for_compare = {
            "ticket_type": pred["ticket_type"],
            "topic": pred["topic"],
            "urgency": pred["urgency"],
        }

        if gold_for_compare == pred_for_compare:
            matches += 1

    return {
        "exact_match_without_tags": matches / len(gold_targets)
    }


def compute_schema_validity(predictions: list[dict]) -> dict:
    valid = 0

    for pred in predictions:
        try:
            TicketRoutingOutput.model_validate(pred)
            valid += 1
        except ValidationError:
            pass

    return {
        "schema_validity": valid / len(predictions)
    }


def main() -> None:
    args = parse_args()

    gold_path = resolve_path(args.gold_path)
    pred_path = resolve_path(args.pred_path)
    report_path = resolve_path(args.report_path)

    gold_rows = read_jsonl(gold_path)
    pred_rows = read_jsonl(pred_path)

    print(f"gold rows: {len(gold_rows)}")
    print(f"pred rows: {len(pred_rows)}")

    if len(gold_rows) != len(pred_rows):
        raise ValueError(
            f"Different number of rows: gold={len(gold_rows)}, pred={len(pred_rows)}"
        )

    gold_targets, predictions = get_targets_and_predictions(gold_rows, pred_rows)

    metrics = {}

    for field in ["ticket_type", "topic", "urgency"]:
        metrics.update(
            compute_metrics(
                gold_targets=gold_targets,
                predictions=predictions,
                field=field
            )
        )

    metrics.update(
        compute_tag_metrics(
            gold_targets=gold_targets,
            predictions=predictions,
        )
    )

    metrics.update(
        compute_exact_match(
            gold_targets=gold_targets,
            predictions=predictions,
        )
    )

    metrics.update(
        compute_exact_match_without_tags(
            gold_targets=gold_targets,
            predictions=predictions,
        )
    )

    metrics.update(
        compute_schema_validity(predictions)
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open('w', encoding='utf8') as f:
        f.write(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved metrics to: {report_path}")

if __name__ == "__main__":
    main()