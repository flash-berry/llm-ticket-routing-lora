# LLM Ticket Routing with LoRA/QLoRA

Production-style pipeline for **structured customer support ticket routing** with strict JSON outputs.

The project compares a deterministic rule baseline, prompt-only LLM baselines, few-shot prompting, and QLoRA fine-tuning of `Qwen/Qwen2.5-0.5B-Instruct` for extracting routing fields from support tickets.

## Project Goal

The goal is to convert an incoming customer support ticket into a validated JSON object:

```json
{
  "ticket_type": "Incident | Request | Problem | Change",
  "topic": "...",
  "urgency": "low | medium | high",
  "tags": ["..."]
}
```

This is a practical structured-output NLP task: the model must not only understand the ticket text, but also follow a closed routing schema that can be consumed by downstream systems.

## Why This Project Matters

Many LLM demos generate free-form text, but production support systems usually need predictable outputs:

- valid JSON;
- schema-constrained fields;
- stable label taxonomy;
- reproducible evaluation;
- error analysis;
- experiment tracking.

This repository demonstrates the full workflow from dataset preparation to baseline comparison, QLoRA fine-tuning, evaluation, MLflow logging, and analysis of model errors.

## Dataset

The project uses an English subset of the `Tobi-Bueck/customer-support-tickets` dataset.

Final processed split:

| Split | Examples |
|---|---:|
| Train | 1,200 |
| Validation | 150 |
| Test | 150 |

Each example is converted into a chat-style supervised fine-tuning format:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Subject: ...\n\nBody: ..."},
    {"role": "assistant", "content": "{...strict JSON...}"}
  ],
  "target": {
    "ticket_type": "...",
    "topic": "...",
    "urgency": "...",
    "tags": ["..."]
  }
}
```

The `target` field is used only for evaluation, not as model input during inference.

## Repository Structure

```text
llm-ticket-routing-lora/
├── configs/                  # Experiment and training configuration files
├── data/                     # Processed train/validation/test JSONL files
├── notebooks/                # EDA, baselines, QLoRA training, error analysis
├── predictions/              # Prediction JSONL files from baselines and LLM runs
├── qlora_manual_outputs/     # QLoRA adapter outputs and generated predictions
├── reports/                  # Metrics JSON files and error analysis reports
├── src/                      # Evaluation, schemas, parsing and utility code
├── requirements.txt          # Python dependencies
└── README.md
```

## Pipeline

The project pipeline consists of the following stages:

1. **Dataset preparation**
   - Load the original support ticket dataset.
   - Filter the English subset.
   - Convert each ticket into chat format.
   - Validate target objects with a Pydantic schema.
   - Save `train.jsonl`, `validation.jsonl`, and `test.jsonl`.

2. **Baselines**
   - Rule-based baseline using keyword heuristics.
   - Zero-shot Qwen prompt baseline.
   - 4-shot Qwen prompt baseline.

3. **Fine-tuning**
   - QLoRA fine-tuning of `Qwen/Qwen2.5-0.5B-Instruct`.
   - 4-bit quantization with LoRA adapters.
   - Token-aware gradient accumulation.
   - Validation loss tracking and early stopping.
   - Best adapter checkpoint saved for inference.

4. **Inference**
   - Generate predictions on the fixed test split.
   - Parse raw model output.
   - Validate JSON and Pydantic schema.
   - Save normalized prediction records as JSONL.

5. **Evaluation**
   - Compare predictions with gold targets by row order.
   - Compute field-level metrics.
   - Compute tag precision, recall and F1.
   - Compute exact match and exact match without tags.
   - Log metrics and reports to MLflow.

6. **Error analysis**
   - Analyze rule baseline failures.
   - Analyze zero-shot and few-shot prompting failures.
   - Compare QLoRA against earlier approaches.

## Models and Experiments

| Experiment | Description |
|---|---|
| `rule_baseline` | Deterministic keyword-based routing baseline |
| `prompt_baseline_qwen` | Zero-shot prompting with Qwen2.5-0.5B-Instruct |
| `prompt_few_shot_qwen` | 4-shot prompting with fixed train examples |
| `qlora_manual` | Manual QLoRA fine-tuning with token-aware training loop |

## Evaluation Metrics

The evaluator reports two types of quality:

### Content quality

- `ticket_type_accuracy`
- `ticket_type_macro_f1`
- `topic_accuracy`
- `topic_macro_f1`
- `urgency_accuracy`
- `urgency_macro_f1`
- `tag_precision`
- `tag_recall`
- `tag_f1`
- `exact_match`
- `exact_match_without_tags`

### Format quality

- `json_validity`
- `schema_validity`

This distinction is important: an output can be syntactically valid JSON but still fail the business schema if it uses a topic outside the allowed taxonomy.

## Results

Evaluation was performed on the same 150-example test split for all experiments.

| Model | Ticket Type Macro-F1 | Topic Macro-F1 | Urgency Macro-F1 | Tag F1 | Exact Match w/o Tags | JSON Validity | Schema Validity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule baseline | 0.3834 | 0.2263 | 0.3185 | 0.1749 | 0.1067 | 1.0000 | 1.0000 |
| Qwen zero-shot | 0.1784 | 0.0286 | 0.2951 | 0.0015 | 0.0000 | 1.0000 | 0.0067 |
| Qwen 4-shot | 0.5154 | 0.1888 | **0.3337** | 0.1562 | 0.0267 | 1.0000 | 0.9067 |
| QLoRA manual | **0.6616** | **0.3260** | 0.3237 | **0.6174** | **0.1733** | 1.0000 | **1.0000** |

### Main result

QLoRA fine-tuning produced the strongest overall model:

- best `ticket_type_macro_f1`;
- best `topic_macro_f1`;
- best `tag_f1`;
- best `exact_match_without_tags`;
- 100% JSON validity;
- 100% schema validity.

The only field where QLoRA did not clearly improve over all baselines was `urgency`, which remained difficult across all approaches.

## Key Findings from Error Analysis

### Rule baseline

The rule baseline is fully valid by construction, but it is limited by keyword coverage. It has low tag recall and often falls back to generic routing decisions.

### Qwen zero-shot

Zero-shot Qwen generates valid JSON, but usually does not respect the closed schema. It often invents free-form `topic` values instead of choosing one of the allowed routing classes.

### Qwen 4-shot

Few-shot prompting improves schema following substantially, but the model still often collapses to generic classes such as `General Inquiry`. This shows that demonstrations help with format, but are not enough to learn the dataset-specific taxonomy.

### QLoRA fine-tuning

QLoRA improves both structure and content quality. It learns the target schema better than prompt-only methods and produces the best results for ticket type, topic and tags. The remaining weak area is urgency classification.

## How to Run

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Run evaluation

Example: evaluate the QLoRA predictions.

```bash
python -m src.evaluate \
  --gold-path data/processed/test.jsonl \
  --pred-path qlora_manual_outputs/predictions/qlora_manual_token_aware_v1_test.jsonl \
  --report-path reports/qlora_manual_metrics.json \
  --experiment-name ticket-routing \
  --run-name qlora_manual
```

Evaluate all main experiments:

```bash
python -m src.evaluate \
  --gold-path data/processed/test.jsonl \
  --pred-path predictions/rule_baseline.jsonl \
  --report-path reports/rule_baseline_metrics.json \
  --experiment-name ticket-routing \
  --run-name rule_baseline

python -m src.evaluate \
  --gold-path data/processed/test.jsonl \
  --pred-path predictions/prompt_baseline_qwen.jsonl \
  --report-path reports/prompt_baseline_qwen_metrics.json \
  --experiment-name ticket-routing \
  --run-name prompt_baseline_qwen

python -m src.evaluate \
  --gold-path data/processed/test.jsonl \
  --pred-path predictions/prompt_few_shot_qwen.jsonl \
  --report-path reports/prompt_few_shot_qwen_metrics.json \
  --experiment-name ticket-routing \
  --run-name prompt_few_shot_qwen

python -m src.evaluate \
  --gold-path data/processed/test.jsonl \
  --pred-path qlora_manual_outputs/predictions/qlora_manual_token_aware_v1_test.jsonl \
  --report-path reports/qlora_manual_metrics.json \
  --experiment-name ticket-routing \
  --run-name qlora_manual
```

### 3. Open MLflow UI

If MLflow is configured to use a local SQLite backend:

```bash
python -m mlflow ui \
  --backend-store-uri sqlite:///C:/Users/Public/mlflow/tracking.db \
  --host 127.0.0.1 \
  --port 5000 \
  --workers 1
```

Then open:

```text
http://127.0.0.1:5000
```

## QLoRA Training Summary

The QLoRA experiment used:

- base model: `Qwen/Qwen2.5-0.5B-Instruct`;
- adapter-based fine-tuning;
- 4-bit quantization;
- token-aware gradient accumulation;
- validation loss for model selection;
- early stopping;
- saved best adapter for inference.

The best validation checkpoint was selected before generating test predictions.

## Reproducibility Notes

- All experiments are evaluated on the same processed test split.
- The evaluator aligns gold and prediction rows by position.
- Prompt baselines and QLoRA predictions are saved as JSONL files.
- Raw model outputs are parsed and validated before scoring.
- MLflow is used for metric tracking and artifact logging.

## Limitations

- The test split contains only 150 examples, so metrics should be interpreted as project-level experimental results rather than production guarantees.
- `urgency` remains difficult across all methods.
- Exact match is very strict because it requires all fields and the full tag set to match simultaneously.
- Tag evaluation uses exact string matching; semantically similar tags with different wording are counted as mismatches.

## Future Work

Potential next experiments:

- train on a larger subset of the dataset;
- use class-balanced sampling for rare labels;
- improve urgency supervision;
- compare Qwen2.5-0.5B with a larger instruction model;
- add constrained decoding or JSON schema-guided generation;
- add a lightweight API or Streamlit demo;
- package the QLoRA inference pipeline into a reproducible CLI.

## Resume Summary

> Built a structured-output support ticket routing system using Qwen2.5-0.5B-Instruct and QLoRA fine-tuning. Implemented dataset preparation, Pydantic schema validation, rule and prompt baselines, token-aware QLoRA training, MLflow experiment tracking, evaluation, and error analysis. Achieved 100% JSON/schema validity and improved ticket routing quality over rule, zero-shot and few-shot baselines.
