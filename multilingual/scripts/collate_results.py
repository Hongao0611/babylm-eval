#!/usr/bin/env python3
"""Collate zeroshot and finetune multilingual results into a single submission JSON.

Output format: {task_name: {subtask_key: accuracy}} — compatible with the leaderboard validator.
Zeroshot tasks use the task name as their single subtask key.
Finetune tasks use the language code as the subtask key (so cross-lingual results are averaged).

Usage:
    python scripts/collate_results.py --model_name gpt2-baseline-BabyLM-2026-Strict
"""

import argparse
import json
from pathlib import Path

# Expected tasks per language — used to warn about partial submissions.
# Keys match the task names as they appear in the collated submission JSON.
EXPECTED_ZEROSHOT = {
    "eng": {"blimp", "hellaswag_en_mubench", "multiblimp_eng", "winogrande_en_mubench", "xstorycloze_en_mubench"},
    "nld": {"blimp_nl", "hellaswag_nl_mubench", "multiblimp_nld", "winogrande_nl_mubench", "xcomps_nl", "xstorycloze_nl_mubench"},
    "zho": {"hellaswag_zh_mubench", "winogrande_zh_mubench", "xcomps_zh", "xstorycloze_zh_mubench", "zhoblimp"},
}

EXPECTED_FINETUNE = {
    "en": {"arc", "belebele", "bmlama", "mnli", "sib200", "truthfulqa", "xnli"},
    "nl": {"arc", "belebele", "bmlama", "include", "mnli", "sib200", "truthfulqa"},
    "zh": {"arc", "belebele", "bmlama", "include", "mnli", "sib200", "truthfulqa", "xnli"},
}

# Fast-eval revisions — scored zeroshot-only at each checkpoint.
REVISIONS = (
    [f"chck_{i}M" for i in range(1, 10)]
    + [f"chck_{i * 10}M" for i in range(1, 10)]
    + [f"chck_{i * 100}M" for i in range(1, 11)]
)


def warn_missing_zeroshot(zeroshot: dict[str, dict[str, float]], label: str) -> None:
    """Warn about missing zeroshot tasks within any language that has at least one task present.

    A language is considered submitted when at least one of its tasks is present.
    Entirely absent languages are silently skipped — incomplete submissions are allowed.
    """
    warned = False
    for lang, expected in EXPECTED_ZEROSHOT.items():
        found = expected & zeroshot.keys()
        if not found:
            continue
        for task in sorted(expected - found):
            print(f"[{label}] Warning: zeroshot task '{task}' ({lang}) is missing — will be scored as 0.")
            warned = True
    if not warned and zeroshot:
        print(f"[{label}] All zeroshot tasks present for every submitted language.")


def warn_missing_finetune(finetune: dict[str, dict[str, float]]) -> None:
    """Warn about missing finetune tasks within any language that has at least one task present."""
    warned = False
    for lang_code, expected in EXPECTED_FINETUNE.items():
        found_benchmarks = {task for task, subtasks in finetune.items() if lang_code in subtasks}
        if not found_benchmarks:
            continue
        for task in sorted(expected - found_benchmarks):
            print(f"[main] Warning: finetune task '{task}' ({lang_code}) is missing — will be scored as 0.")
            warned = True
    if not warned and finetune:
        print("[main] All finetune tasks present for every submitted language.")


def parse_zeroshot(results: dict) -> dict[str, dict[str, float]]:
    """Extract indent-1 tasks from a lm-eval results dict.

    Indentation in the alias field encodes depth:
      indent 0 → language group aggregate (e.g. zeroshot_eng)  → skip
      indent 1 → task row (e.g. blimp, hellaswag_en_mubench)   → keep
      indent 2 → subtask row (e.g. blimp_adjunct_island)        → skip (already folded into indent-1)
    """
    out: dict[str, dict[str, float]] = {}
    for task_name, task_data in results.items():
        alias = task_data.get("alias", task_name)
        indent = len(alias) - len(alias.lstrip())
        acc = task_data.get("acc,none")
        if indent == 1 and acc is not None:
            out[task_name] = {task_name: acc}
    return out


def load_zeroshot(results_dir: Path, model_name: str) -> dict[str, dict[str, float]]:
    """Load all results_*.json files from the model's folder in results/.

    Folder name can be org__model or just model; matches on the part after the last '__'.
    Multiple JSON files in the same folder (one per language run) are merged.
    """
    if not results_dir.exists():
        print(f"Warning: no zeroshot results folder found at {results_dir}")
        return {}

    matches = [d for d in results_dir.iterdir() if d.is_dir() and d.name.split("__")[-1] == model_name]
    if not matches:
        print(f"Warning: no zeroshot results folder found for '{model_name}' in {results_dir}")
        return {}

    combined: dict[str, dict[str, float]] = {}
    for folder in matches:
        json_files = sorted(folder.glob("results_*.json"))
        if not json_files:
            print(f"Warning: no results_*.json found in {folder}")
            continue
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
            combined.update(parse_zeroshot(data["results"]))

    return combined


def load_finetune(finetune_dir: Path, model_name: str) -> dict[str, dict[str, float]]:
    """Load eval_results.json files from finetune/results/{model_name}/{lang}/{task}/.

    Returns {task: {lang: eval_accuracy}}.
    """
    model_dir = finetune_dir / model_name
    if not model_dir.exists():
        print(f"Warning: no finetune results folder found for '{model_name}' in {finetune_dir}")
        return {}

    out: dict[str, dict[str, float]] = {}
    for result_file in sorted(model_dir.glob("*/*/eval_results.json")):
        lang = result_file.parts[-3]
        task = result_file.parts[-2]
        with open(result_file) as f:
            data = json.load(f)
        acc = data.get("eval_accuracy")
        if acc is None:
            continue
        out.setdefault(task, {})[lang] = acc

    return out


def load_zeroshot_raw(results_dir: Path, model_name: str) -> dict:
    """Merge the raw lm-eval 'results' dicts from all results_*.json files for a model."""
    if not results_dir.exists():
        return {}

    matches = [d for d in results_dir.iterdir() if d.is_dir() and d.name.split("__")[-1] == model_name]
    if not matches:
        return {}

    merged: dict = {}
    for folder in matches:
        for json_file in sorted(folder.glob("results_*.json")):
            with open(json_file) as f:
                data = json.load(f)
            merged.update(data.get("results", {}))
    return merged


def load_finetune_predictions(finetune_dir: Path, model_name: str) -> dict[str, list[dict]]:
    """Load predictions.txt files from finetune/results/{model_name}/{lang}/{task}/.

    Returns {"{task}_{lang}": [{"index": int, "prediction": str}, ...]}.
    """
    model_dir = finetune_dir / model_name
    if not model_dir.exists():
        return {}

    out: dict[str, list[dict]] = {}
    for pred_file in sorted(model_dir.glob("*/*/predictions.txt")):
        lang = pred_file.parts[-3]
        task = pred_file.parts[-2]
        key = f"{task}_{lang}"
        rows = []
        with open(pred_file) as f:
            header = f.readline()  # skip "index\tprediction" header
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                rows.append({"index": int(parts[0]), "prediction": parts[1] if len(parts) > 1 else ""})
        out[key] = rows
    return out


def main():
    parser = argparse.ArgumentParser(description="Collate multilingual evaluation results into a submission JSON.")
    parser.add_argument("--model_name", required=True,
                        help="Model folder name to match. For zeroshot results the part after '__' is matched; "
                             "for finetune results the folder name is matched directly.")
    parser.add_argument("--output",
                        help="Output JSON path (default: <model_name>_submission.json in cwd)")
    parser.add_argument("--output_predictions",
                        help="Output predictions JSON path (default: <model_name>_predictions.json in cwd)")
    parser.add_argument("--fast", action="store_true",
                        help="Also collate per-revision zeroshot accuracy summaries from results/<revision>/.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent  # multilingual/
    zeroshot = load_zeroshot(root / "results" / "main", args.model_name)
    finetune = load_finetune(root / "finetune" / "results", args.model_name)

    warn_missing_zeroshot(zeroshot, label="main")
    warn_missing_finetune(finetune)

    fast_summaries: list[dict[str, float]] = []
    if args.fast:
        for rev in REVISIONS:
            rev_zeroshot = load_zeroshot(root / "results" / rev, args.model_name)
            warn_missing_zeroshot(rev_zeroshot, label=rev)
            fast_summaries.append({task: subtasks[task] for task, subtasks in rev_zeroshot.items()})

    combined = {**zeroshot, **finetune}

    if not combined:
        print("No results found — nothing written.")
        return

    output_path = Path(args.output) if args.output else Path(f"{args.model_name}_submission.json")
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Wrote {len(combined)} task(s) to {output_path}")
    for task, subtasks in combined.items():
        avg = sum(subtasks.values()) / len(subtasks)
        n = len(subtasks)
        print(f"  {task}: {avg:.4f}  ({n} subtask{'s' if n > 1 else ''})")

    # --- predictions file ---
    zeroshot_raw = load_zeroshot_raw(root / "results" / "main", args.model_name)
    finetune_preds = load_finetune_predictions(root / "finetune" / "results", args.model_name)

    if zeroshot_raw or finetune_preds or fast_summaries:
        predictions = {
            "zeroshot": zeroshot_raw,
            "finetune": finetune_preds,
        }
        if fast_summaries:
            predictions["fast_eval_results"] = fast_summaries
        pred_path = Path(args.output_predictions) if args.output_predictions else Path(f"{args.model_name}_predictions.json")
        with open(pred_path, "w") as f:
            json.dump(predictions, f, indent=2)
        msg = (f"Wrote predictions file to {pred_path} "
               f"({len(zeroshot_raw)} zeroshot tasks, {len(finetune_preds)} finetune task-lang pairs")
        if fast_summaries:
            msg += f", {len(fast_summaries)} fast-eval revisions"
        msg += ")"
        print(msg)


if __name__ == "__main__":
    main()
