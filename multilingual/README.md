# Multilingual Track Evaluation

This directory contains the evaluation pipeline for the **multilingual track** of the 2026 BabyLM Challenge. It supports English (eng), Dutch (nld), and Chinese (zho), and is built on top of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (zero-shot) and a custom fine-tuning script (finetune tasks).

Incomplete evaluation is explicitly allowed: you may submit results for only the language(s) your model covers. Missing tasks are set to 0 and factored into the average scores on the leaderboard.

---

## Tasks

Evaluation is divided into **zero-shot** tasks (scored directly by the LM's log-probabilities) and **finetune** tasks (trained per-task with a small classification head).

### Zero-shot

| Language | Tasks |
|----------|-------|
| English  | BLiMP, HellaSwag, MultiBLiMP, Winogrande, XStoryCloze |
| Dutch    | BLiMP-NL, HellaSwag, MultiBLiMP, Winogrande, XCOMPS, XStoryCloze |
| Chinese  | HellaSwag, Winogrande, XCOMPS, XStoryCloze, ZhoBLiMP |

Custom task definitions are in `tasks/`.

### Finetune

| Language | Tasks |
|----------|-------|
| English  | ARC, Belebele, BMLama, MNLI, SIB-200, TruthfulQA, XNLI |
| Dutch    | ARC, Belebele, BMLama, INCLUDE, MNLI, SIB-200, TruthfulQA |
| Chinese  | ARC, Belebele, BMLama, INCLUDE, MNLI, SIB-200, TruthfulQA, XNLI |

---

## Running Evaluation

### Zero-shot

Evaluate a model across all three languages:

```bash
bash scripts/zeroshot_model.sh --model_name YOUR_MODEL
```

To restrict to specific languages:

```bash
bash scripts/zeroshot_model.sh --model_name YOUR_MODEL --langs "eng nld"
```

Results are written to `results/<org__model>/results_<timestamp>.json`.

### Finetune

```bash
bash scripts/finetune_model.sh --model_name YOUR_MODEL --langs "eng nld zho"
```

Optional hyperparameter flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--lr` | `5e-5` | Learning rate |
| `--bsz` | `64` | Batch size |
| `--max_epochs` | `10` | Max training epochs |
| `--patience` | `3` | Early stopping patience |
| `--seed` | `12` | Random seed |

Results are written to `finetune/results/<model>/<lang>/<task>/`.

---

## Collating Results for Submission

Once evaluation is complete, run `scripts/collate_results.py` to produce the submission files:

```bash
python scripts/collate_results.py --model_name YOUR_MODEL
```

This produces **two output files**:

### `<model_name>_submission.json` — leaderboard scores file

This is the file you upload to the leaderboard. It contains pre-computed accuracy scores in the format expected by the leaderboard validator:

```json
{
  "blimp":              {"blimp": 0.734},
  "hellaswag_en_mubench": {"hellaswag_en_mubench": 0.265},
  "arc":                {"en": 0.248, "nl": 0.244}
}
```

Zeroshot tasks use the task name as their single key. Finetune tasks use the language code as the key so results from multiple languages are grouped under one benchmark name.

### `<model_name>_predictions.json` — raw predictions file

This file is also uploaded to the leaderboard alongside the scores file. It is not used for scoring — it allows organizers to verify submissions if needed. It has two top-level keys:

- **`"zeroshot"`** — the raw lm-eval `results` dict merged across all `results_*.json` files (includes individual subtask rows such as BLiMP paradigms)
- **`"finetune"`** — keyed as `"{task}_{lang}"` (e.g. `"arc_nl"`), each holding the list of per-example predictions from `predictions.txt`

```json
{
  "zeroshot": {
    "blimp": {"acc,none": 0.734, ...},
    "blimp_adjunct_island": {"acc,none": 0.812, ...}
  },
  "finetune": {
    "arc_en": [{"index": 0, "prediction": "C"}, ...],
    "arc_nl": [{"index": 0, "prediction": "B"}, ...]
  }
}
```

### Missing task warnings

Before writing anything, the script checks whether the submission is complete for each submitted language. If a language has at least one result but is missing other expected tasks, a warning is printed for each:

```
Warning: zeroshot task 'multiblimp_eng' (eng) is missing — will be scored as 0.
Warning: finetune task 'xnli' (en) is missing — will be scored as 0.
```

Languages with no results at all are silently skipped (intentional partial submissions are fine). If everything is present you will see:

```
All tasks present for every submitted language.
```

### Custom output paths

```bash
python scripts/collate_results.py \
    --model_name YOUR_MODEL \
    --output path/to/submission.json \
    --output_predictions path/to/predictions.json
```

---

## Viewing Results Locally

Two helper scripts let you inspect results without uploading anything.

**Zero-shot results table** (markdown, grouped by language):

```bash
python scripts/print_results_table.py --results_dir results/
```

**Finetune results table** (markdown, grouped by language):

```bash
python scripts/print_finetune_results.py
```

---

## Submitting to the Leaderboard

Upload both output files from `collate_results.py` on the leaderboard submission page:

- **Results file** (`*_submission.json`) — used for scoring
- **Predictions file** (`*_predictions.json`) — required for the multilingual track, used by organizers to verify submissions

The leaderboard is live at: [![Leaderboard](https://img.shields.io/badge/🤗-Leaderboard-yellow)](https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026)
