import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prediction_utils import parse_and_validate_response


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prompt-only LLM baseline for ticket routing."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "prompt_baseline_qwen.yaml",
        help="Path to YAML config file.",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    model.eval()

    return model, tokenizer


def generate_response(
        row: dict,
        model,
        tokenizer,
        max_new_tokens: int,
        temperature: float,
) -> str:
    messages = row["messages"][:2]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }

    if temperature > 0.0:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

    generated_ids = outputs_ids[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)

    config = load_config(config_path)

    input_path = resolve_path(Path(config["data"]["input_path"]))
    pred_path = resolve_path(Path(config["output"]["pred_path"]))
    max_examples = config["data"].get("max_examples", None)

    rows = read_jsonl(input_path)

    if max_examples is not None:
        rows = rows[:max_examples]

    print(f"Config: {config_path}")
    print(f"Input path: {input_path}")
    print(f"Prediction path: {pred_path}")
    print(f"Rows to process: {len(rows)}")
    print(f"Model: {config['model']['name']}")

    model_name = config["model"]["name"]
    max_new_tokens = int(config["model"]["max_new_tokens"])
    temperature = float(config["model"]["temperature"])

    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_name)
    print("Model loaded.")

    predictions = []

    for index, row in enumerate(rows, start=1):
        raw_response = generate_response(
            row=row,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        result = parse_and_validate_response(raw_response)
        predictions.append(result)

        print(
            f"[{index}/{len(rows)}] "
            f"json_valid={result['json_valid']}, "
            f"schema_valid={result['schema_valid']}"
        )

    save_jsonl(predictions, pred_path)

    print("=" * 80)
    print(f"Saved predictions to: {pred_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()