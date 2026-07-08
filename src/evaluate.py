import argparse
import mlflow
import json
from pathlib import Path
from mlflow.tracking import MlflowClient

from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GOLD_PATH = PROJECT_ROOT / "data" / "processed" / "test.jsonl"
DEFAULT_PRED_PATH = PROJECT_ROOT / "predictions" / "rule_baseline.jsonl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "rule_baseline_metrics.json"

MLFLOW_ROOT = Path(r"C:\Users\Public\mlflow")
MLFLOW_DB_PATH = MLFLOW_ROOT / "tracking.db"
MLFLOW_ARTIFACTS_PATH = MLFLOW_ROOT / "artifacts"


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
        help="Path to report JSON file.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ticket-routing",
        help="MLflow experiment name.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="rule_baseline",
        help="MLflow run name.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional limit for gold examples, useful for debug runs.",
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


def make_invalid_prediction() -> dict:
    return {
        "ticket_type": "__INVALID__",
        "topic": "__INVALID__",
        "urgency": "__INVALID__",
        "tags": [],
    }


def get_content_prediction(pred_row: dict) -> dict:
    parsed_response = pred_row.get("parsed_response")

    if not isinstance(parsed_response, dict):
        return make_invalid_prediction()

    ticket_type = parsed_response.get("ticket_type", "__INVALID__")
    topic = parsed_response.get("topic", "__INVALID__")
    urgency = parsed_response.get("urgency", "__INVALID__")
    tags = parsed_response.get("tags", [])

    if not isinstance(ticket_type, str):
        ticket_type = "__INVALID__"

    if not isinstance(topic, str):
        topic = "__INVALID__"

    if not isinstance(urgency, str):
        urgency = "__INVALID__"

    if not isinstance(tags, list):
        tags = []

    clean_tags = [
        tag.strip()
        for tag in tags
        if isinstance(tag, str) and tag.strip()
    ]

    return {
        "ticket_type": ticket_type,
        "topic": topic,
        "urgency": urgency,
        "tags": clean_tags,
    }


def get_targets_and_predictions(
    gold_rows: list[dict],
    pred_rows: list[dict],
) -> tuple[list[dict], list[dict], list[bool], list[bool]]:
    gold_targets = []
    predictions = []
    json_valid_flags = []
    schema_valid_flags = []

    for gold_row, pred_row in zip(gold_rows, pred_rows):
        gold_targets.append(gold_row["target"])

        prediction = get_content_prediction(pred_row)
        predictions.append(prediction)

        json_valid_flags.append(
            bool(pred_row.get("json_valid", False))
        )
        schema_valid_flags.append(
            bool(pred_row.get("schema_valid", False))
        )

    return gold_targets, predictions, json_valid_flags, schema_valid_flags


def compute_metrics(
        gold_targets: list[dict],
        predictions: list[dict],
        field: str
) -> dict:
    y_true = [row[field] for row in gold_targets]
    y_pred = [row[field] for row in predictions]

    observed_labels = sorted(set(y_true))

    return {
        f"{field}_accuracy": accuracy_score(y_true, y_pred),
        f"{field}_macro_f1": f1_score(
            y_true,
            y_pred,
            labels=observed_labels,
            average="macro",
            zero_division=0,
        ),
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


def compute_json_validity(json_valid_flags: list[bool]) -> dict:
    return {
        "json_validity": sum(json_valid_flags) / len(json_valid_flags)
    }


def compute_schema_validity(schema_valid_flags: list[bool]) -> dict:
    return {
        "schema_validity": sum(schema_valid_flags) / len(schema_valid_flags)
    }


def configure_mlflow(experiment_name: str) -> None:
    MLFLOW_ROOT.mkdir(parents=True, exist_ok=True)
    MLFLOW_ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
    )

    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        client.create_experiment(
            name=experiment_name,
            artifact_location=MLFLOW_ARTIFACTS_PATH.resolve().as_uri(),
        )

    mlflow.set_experiment(experiment_name)


def main() -> None:
    args = parse_args()

    gold_path = resolve_path(args.gold_path)
    pred_path = resolve_path(args.pred_path)
    report_path = resolve_path(args.report_path)

    gold_rows = read_jsonl(gold_path)
    pred_rows = read_jsonl(pred_path)

    if args.max_examples is not None:
        gold_rows = gold_rows[:args.max_examples]
        pred_rows = pred_rows[:args.max_examples]

    print(f"gold rows: {len(gold_rows)}")
    print(f"pred rows: {len(pred_rows)}")

    if len(gold_rows) != len(pred_rows):
        raise ValueError(
            f"Different number of rows: gold={len(gold_rows)}, pred={len(pred_rows)}"
        )

    gold_targets, predictions, json_valid_flags, schema_valid_flags = (
        get_targets_and_predictions(gold_rows, pred_rows)
    )

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

    metrics.update(compute_json_validity(json_valid_flags))
    metrics.update(compute_schema_validity(schema_valid_flags))

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open('w', encoding='utf8') as f:
        f.write(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved metrics to: {report_path}")

    configure_mlflow(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("gold_path", str(gold_path))
        mlflow.log_param("pred_path", str(pred_path))
        mlflow.log_param("report_path", str(report_path))

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.log_artifact(str(report_path))

if __name__ == "__main__":
    main()