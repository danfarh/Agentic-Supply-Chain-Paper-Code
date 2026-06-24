#!/usr/bin/env python3
"""
Evaluate ablation-study outputs against the structured reference answer key.

This script is intended to live in:
    src/evaluation/Ablation/run_ablations_evaluation.py

It evaluates already-generated ablation answer files, for example:
    src/evaluation/Ablation/llm_only/generated_answers_structured.json
    src/evaluation/Ablation/without_ethics/generated_answers_structured.json
    src/evaluation/Ablation/without_rag/generated_answers_structured.json

For each ablation folder, it calls the same direct evaluator used by
src/evaluation/run_evaluation_full.py, passing:
    --generated <variant generated_answers_structured.json>
    --answer-key <reference answer key>
    --out-dir <variant folder>/outputs

Outputs are written inside an outputs folder for each ablation variant:
    <variant>/outputs/evaluation_results.json
    <variant>/outputs/evaluation_results.csv
    <variant>/outputs/evaluation_summary.json
    <variant>/outputs/evaluation_summary.csv
    <variant>/outputs/rows_with_contexts.csv

Combined comparison files are also written to:
    Ablation/outputs/ablation_evaluation_comparison.json
    Ablation/outputs/ablation_evaluation_comparison.csv

No prebuilt ragas_dataset_all_28_with_generated_contexts.json file is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Defaults and aliases
# -----------------------------------------------------------------------------

DEFAULT_VARIANTS = ["llm_only", "without_ethics", "without_rag"]

# Explicit thresholds used for all ablation evaluations unless overridden from CLI.
# These are passed to src/evaluation/run_evaluation_full.py for every variant,
# so llm_only / without_ethics / without_rag are judged with identical cutoffs.
DEFAULT_PASS_THRESHOLD = 0.75
DEFAULT_PARTIAL_THRESHOLD = 0.50

VARIANT_ALIASES: Dict[str, Sequence[str]] = {
    "llm_only": (
        "llm_only",
        "ll_only",
        "llm-only",
        "llm only",
        "LLM-only",
        "LLM_only",
        "baseline_llm_only",
    ),
    "without_ethics": (
        "without_ethics",
        "without ethics",
        "without-ethics",
        "mas_without_ethics",
        "MAS_without_ethics",
        "mas-without-ethics",
        "no_ethics",
        "no-ethics",
    ),
    "without_rag": (
        "without_rag",
        "withput_rag",  # common typo
        "without rag",
        "without-rag",
        "mas_without_rag",
        "MAS_without_rag",
        "mas-without-rag",
        "no_rag",
        "no-rag",
    ),
}

REFERENCE_CANDIDATE_NAMES = (
    "structured_reference_answer_key.json",
    "structured_answer_key_MAS_evaluation_final.json",
    "structured_answer_key_USER_REFERENCE_FINAL.json",
    "structured_answer_key_all_28_FINAL.json",
)

GENERATED_CANDIDATE_NAMES = (
    "generated_answers_structured.json",
    "generated_answers.json",
    "answers_generated.json",
)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fields is None:
        seen: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.append(key)
        fields = seen
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: flatten(row.get(field)) for field in fields})


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        try:
            key = str(p.resolve()) if p.exists() else str(p)
        except Exception:
            key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def evaluation_dir_from_ablation_dir(ablation_dir: Path) -> Path:
    # Ablation folder should be src/evaluation/Ablation.
    return ablation_dir.resolve().parent


def project_root_from_ablation_dir(ablation_dir: Path) -> Path:
    # src/evaluation/Ablation -> project root is three parents up.
    # If structure differs, this is still only used for PYTHONPATH.
    p = ablation_dir.resolve()
    try:
        return p.parents[2]
    except IndexError:
        return p.parent


def resolve_existing_path(raw: str | Path, base_dirs: Sequence[Path], extra_names: Sequence[str] = ()) -> Path:
    """Resolve a file path using a set of base dirs and fallback names."""
    raw_path = Path(raw).expanduser()
    candidates: List[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(raw_path)
        for base in base_dirs:
            candidates.append(base / raw_path)

    names = [raw_path.name] + [n for n in extra_names if n]
    for name in names:
        for base in base_dirs:
            candidates.append(base / name)
            candidates.append(base / "logs" / name)

    candidates = unique_paths(candidates)
    for p in candidates:
        if p.exists() and p.is_file():
            return p.resolve()

    checked = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(f"File not found. Checked:\n{checked}")


# -----------------------------------------------------------------------------
# Variant discovery
# -----------------------------------------------------------------------------


@dataclass
class VariantTarget:
    canonical_name: str
    folder: Path
    generated_json: Path
    output_dir: Path


def candidate_variant_folders(ablation_dir: Path, canonical_name: str) -> List[Path]:
    aliases = VARIANT_ALIASES.get(canonical_name, (canonical_name,))
    candidates: List[Path] = []

    for alias in aliases:
        candidates.append(ablation_dir / alias)

    # Common structure from run_ablations.py: results/<run_id>/<variant>/
    results_dir = ablation_dir / "results"
    if results_dir.exists():
        for run_dir in sorted(results_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            for alias in aliases:
                candidates.append(run_dir / alias)

    return unique_paths(candidates)


def find_generated_json_in_folder(folder: Path, generated_filename: str) -> Optional[Path]:
    preferred = [
        folder / generated_filename,
        folder / "logs" / generated_filename,
        folder / "outputs" / generated_filename,
    ]
    for name in GENERATED_CANDIDATE_NAMES:
        preferred.extend([
            folder / name,
            folder / "logs" / name,
            folder / "outputs" / name,
        ])
    for p in unique_paths(preferred):
        if p.exists() and p.is_file():
            return p.resolve()

    # Last-resort recursive search. Prefer shorter/deeper-stable paths.
    matches = [p for p in folder.rglob(generated_filename) if p.is_file()]
    if not matches:
        for name in GENERATED_CANDIDATE_NAMES:
            matches.extend([p for p in folder.rglob(name) if p.is_file()])
    if matches:
        matches = sorted(set(matches), key=lambda p: (len(p.parts), str(p)))
        return matches[0].resolve()
    return None


def resolve_variant_target(
    ablation_dir: Path,
    canonical_name: str,
    generated_filename: str,
    out_subdir: str,
) -> VariantTarget:
    attempted: List[Path] = []
    for folder in candidate_variant_folders(ablation_dir, canonical_name):
        attempted.append(folder)
        if not folder.exists() or not folder.is_dir():
            continue
        gen = find_generated_json_in_folder(folder, generated_filename)
        if gen is not None:
            output_dir = folder / out_subdir if out_subdir else folder
            return VariantTarget(canonical_name, folder.resolve(), gen, output_dir.resolve())

    checked = "\n".join(f"- {p}" for p in attempted)
    raise FileNotFoundError(
        f"Could not find generated answers for variant '{canonical_name}'.\n"
        f"Looked in these folders:\n{checked}\n\n"
        f"Expected a file named '{generated_filename}' or one of: {', '.join(GENERATED_CANDIDATE_NAMES)}"
    )


# -----------------------------------------------------------------------------
# Evaluator execution
# -----------------------------------------------------------------------------


def build_evaluator_command(
    python_executable: str,
    evaluator_path: Path,
    generated_path: Path,
    answer_key_path: Path,
    out_dir: Path,
    pass_threshold: Optional[float],
    partial_threshold: Optional[float],
    use_official_ragas: bool,
) -> List[str]:
    cmd = [
        python_executable,
        str(evaluator_path),
        "--generated",
        str(generated_path),
        "--answer-key",
        str(answer_key_path),
        "--out-dir",
        str(out_dir),
    ]
    if pass_threshold is not None:
        cmd.extend(["--pass-threshold", str(pass_threshold)])
    if partial_threshold is not None:
        cmd.extend(["--partial-threshold", str(partial_threshold)])
    if use_official_ragas:
        cmd.append("--use-official-ragas")
    return cmd


def run_evaluator_for_variant(
    target: VariantTarget,
    evaluator_path: Path,
    answer_key_path: Path,
    python_executable: str,
    pass_threshold: Optional[float],
    partial_threshold: Optional[float],
    use_official_ragas: bool,
    dry_run: bool,
    env: Dict[str, str],
) -> Dict[str, Any]:
    target.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_evaluator_command(
        python_executable=python_executable,
        evaluator_path=evaluator_path,
        generated_path=target.generated_json,
        answer_key_path=answer_key_path,
        out_dir=target.output_dir,
        pass_threshold=pass_threshold,
        partial_threshold=partial_threshold,
        use_official_ragas=use_official_ragas,
    )

    print("\n" + "=" * 88)
    print(f"Evaluating variant: {target.canonical_name}")
    print(f"Folder:    {target.folder}")
    print(f"Generated: {target.generated_json}")
    print(f"Out dir:   {target.output_dir}")
    print("Command:   " + " ".join(cmd))

    record: Dict[str, Any] = {
        "variant": target.canonical_name,
        "variant_folder": str(target.folder),
        "generated_json": str(target.generated_json),
        "output_dir": str(target.output_dir),
        "command": cmd,
        "started_utc": now_utc_iso(),
        "finished_utc": None,
        "returncode": None,
        "success": False,
        "stdout": "",
        "stderr": "",
    }

    if dry_run:
        record.update({"success": True, "returncode": 0, "finished_utc": now_utc_iso(), "dry_run": True})
        return record

    completed = subprocess.run(
        cmd,
        cwd=str(evaluator_path.parent),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    record["finished_utc"] = now_utc_iso()
    record["returncode"] = completed.returncode
    record["success"] = completed.returncode == 0
    record["stdout"] = completed.stdout
    record["stderr"] = completed.stderr

    if completed.stdout.strip():
        print(completed.stdout)
    if completed.stderr.strip():
        print("STDERR:", completed.stderr, file=sys.stderr)

    if completed.returncode != 0:
        raise RuntimeError(
            f"Evaluator failed for variant '{target.canonical_name}' with return code {completed.returncode}.\n"
            f"STDERR:\n{completed.stderr}"
        )

    return record


# -----------------------------------------------------------------------------
# Summary aggregation
# -----------------------------------------------------------------------------


def load_variant_summary(target: VariantTarget) -> Dict[str, Any]:
    summary_path = target.output_dir / "evaluation_summary.json"
    results_path = target.output_dir / "evaluation_results.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary was not created: {summary_path}")
    obj = read_json(summary_path)
    return {
        "summary_path": str(summary_path),
        "results_path": str(results_path),
        "summary_json": obj,
    }


def extract_main_summary(summary_obj: Dict[str, Any]) -> Dict[str, Any]:
    summary = summary_obj.get("summary", {}) if isinstance(summary_obj, dict) else {}
    # run_evaluation_full.py direct version uses all_rows and final_accuracy_rows_only.
    main = summary.get("final_accuracy_rows_only") or summary.get("all_rows") or summary.get("all_28_rows") or {}
    rag = summary.get("rows_with_retrieved_contexts") or summary.get("rag_rows_with_retrieved_contexts") or {}
    return {
        "n": main.get("n"),
        "pass": main.get("pass"),
        "partial": main.get("partial"),
        "fail": main.get("fail"),
        "missing": main.get("missing"),
        "strict_pass_rate": main.get("strict_pass_rate"),
        "pass_or_partial_rate": main.get("pass_or_partial_rate"),
        "mean_answer_correctness": main.get("mean_answer_correctness"),
        "mean_answer_relevancy": main.get("mean_answer_relevancy"),
        "rows_with_contexts": main.get("rows_with_contexts"),
        "rag_n": rag.get("n"),
        "rag_mean_composite": rag.get("mean_rag_composite"),
        "rag_mean_faithfulness": rag.get("mean_faithfulness"),
        "rag_mean_context_precision": rag.get("mean_context_precision"),
        "rag_mean_context_recall": rag.get("mean_context_recall"),
    }


def build_comparison_rows(targets: Sequence[VariantTarget]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    full: Dict[str, Any] = {}
    for target in targets:
        summary_bundle = load_variant_summary(target)
        summary_obj = summary_bundle["summary_json"]
        main = extract_main_summary(summary_obj)
        row = {
            "variant": target.canonical_name,
            "variant_folder": str(target.folder),
            "generated_json": str(target.generated_json),
            "output_dir": str(target.output_dir),
            "summary_path": summary_bundle["summary_path"],
            **main,
        }
        rows.append(row)
        full[target.canonical_name] = {
            "paths": {
                "variant_folder": str(target.folder),
                "generated_json": str(target.generated_json),
                "output_dir": str(target.output_dir),
                "summary_path": summary_bundle["summary_path"],
                "results_path": summary_bundle["results_path"],
            },
            "summary": summary_obj,
        }
    return rows, full


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    default_ablation_dir = script_dir()
    default_eval_dir = evaluation_dir_from_ablation_dir(default_ablation_dir)

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ablation generated_answers_structured.json files directly against "
            "a structured reference answer key, using src/evaluation/run_evaluation_full.py."
        )
    )
    parser.add_argument(
        "--ablation-dir",
        default=str(default_ablation_dir),
        help="Path to src/evaluation/Ablation. Default: folder containing this script.",
    )
    parser.add_argument(
        "--evaluator",
        default=str(default_eval_dir / "run_evaluation_full.py"),
        help="Path to direct evaluator script. Default: ../run_evaluation_full.py",
    )
    parser.add_argument(
        "--answer-key",
        "--reference",
        dest="answer_key",
        default=str(default_eval_dir / "logs" / "structured_reference_answer_key.json"),
        help="Path to structured reference answer key JSON.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=DEFAULT_VARIANTS,
        help="Canonical variants to evaluate. Default: llm_only without_ethics without_rag",
    )
    parser.add_argument(
        "--generated-filename",
        default="generated_answers_structured.json",
        help="Generated answers filename to look for inside each variant folder.",
    )
    parser.add_argument(
        "--out-subdir",
        default="outputs",
        help=(
            "Subdirectory inside each variant folder for outputs. Default: outputs. "
            "This writes evaluation_results/summary into <variant>/outputs/."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to call run_evaluation_full.py. Default: current Python.",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=DEFAULT_PASS_THRESHOLD,
        help=(
            "PASS threshold for final_score_for_accuracy. "
            f"Default: {DEFAULT_PASS_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--partial-threshold",
        type=float,
        default=DEFAULT_PARTIAL_THRESHOLD,
        help=(
            "PARTIAL threshold for final_score_for_accuracy. "
            f"Default: {DEFAULT_PARTIAL_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--use-official-ragas",
        action="store_true",
        help="Pass --use-official-ragas to the evaluator for rows with retrieved contexts.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip variants whose generated_answers file is missing instead of failing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run but do not evaluate.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    ablation_dir = Path(args.ablation_dir).expanduser().resolve()
    evaluation_dir = evaluation_dir_from_ablation_dir(ablation_dir)
    project_root = project_root_from_ablation_dir(ablation_dir)

    evaluator_path = resolve_existing_path(
        args.evaluator,
        base_dirs=(Path.cwd(), evaluation_dir, ablation_dir),
        extra_names=("run_evaluation_full.py",),
    )
    answer_key_path = resolve_existing_path(
        args.answer_key,
        base_dirs=(Path.cwd(), evaluation_dir, evaluation_dir / "logs", ablation_dir),
        extra_names=REFERENCE_CANDIDATE_NAMES,
    )

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(project_root), str(evaluation_dir.parent), existing_pythonpath]
    env["PYTHONPATH"] = os.pathsep.join([p for p in pythonpath_parts if p])

    print("Ablation evaluation")
    print(f"Ablation dir: {ablation_dir}")
    print(f"Evaluator:    {evaluator_path}")
    print(f"Answer key:   {answer_key_path}")
    print(f"Project root: {project_root}")
    print(f"PASS threshold: {args.pass_threshold}")
    print(f"PARTIAL threshold: {args.partial_threshold}")

    ablation_outputs_dir = ablation_dir / "outputs"
    ablation_outputs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ablation-level outputs: {ablation_outputs_dir}")

    targets: List[VariantTarget] = []
    skipped: List[Dict[str, str]] = []
    for variant in args.variants:
        canonical = variant.strip()
        if canonical not in VARIANT_ALIASES:
            # Allow user to pass an actual folder name as a one-off canonical variant.
            VARIANT_ALIASES[canonical] = (canonical,)
        try:
            target = resolve_variant_target(
                ablation_dir=ablation_dir,
                canonical_name=canonical,
                generated_filename=args.generated_filename,
                out_subdir=args.out_subdir,
            )
            targets.append(target)
        except FileNotFoundError as exc:
            if args.skip_missing:
                skipped.append({"variant": canonical, "reason": str(exc)})
                print(f"Skipping missing variant '{canonical}': {exc}")
            else:
                raise

    run_records: List[Dict[str, Any]] = []
    for target in targets:
        record = run_evaluator_for_variant(
            target=target,
            evaluator_path=evaluator_path,
            answer_key_path=answer_key_path,
            python_executable=args.python,
            pass_threshold=args.pass_threshold,
            partial_threshold=args.partial_threshold,
            use_official_ragas=args.use_official_ragas,
            dry_run=args.dry_run,
            env=env,
        )
        run_records.append(record)

    run_log = {
        "created_utc": now_utc_iso(),
        "ablation_dir": str(ablation_dir),
        "evaluator_path": str(evaluator_path),
        "answer_key_path": str(answer_key_path),
        "variants_requested": args.variants,
        "variants_evaluated": [t.canonical_name for t in targets],
        "skipped": skipped,
        "dry_run": bool(args.dry_run),
        "pass_threshold": args.pass_threshold,
        "partial_threshold": args.partial_threshold,
        "runs": run_records,
    }
    write_json(ablation_outputs_dir / "ablation_evaluation_run_log.json", run_log)

    if not args.dry_run and targets:
        comparison_rows, full = build_comparison_rows(targets)
        comparison = {
            "created_utc": now_utc_iso(),
            "answer_key_path": str(answer_key_path),
            "pass_threshold": args.pass_threshold,
            "partial_threshold": args.partial_threshold,
            "comparison_rows": comparison_rows,
            "full_summaries": full,
        }
        write_json(ablation_outputs_dir / "ablation_evaluation_comparison.json", comparison)
        write_csv(ablation_outputs_dir / "ablation_evaluation_comparison.csv", comparison_rows)

        print("\n" + "=" * 88)
        print("Ablation comparison summary")
        for row in comparison_rows:
            print(
                f"{row['variant']}: "
                f"PASS={row.get('pass')} PARTIAL={row.get('partial')} FAIL={row.get('fail')} "
                f"strict={row.get('strict_pass_rate')} pass_or_partial={row.get('pass_or_partial_rate')} "
                f"contexts={row.get('rows_with_contexts')}"
            )
        print(f"\nComparison JSON: {ablation_outputs_dir / 'ablation_evaluation_comparison.json'}")
        print(f"Comparison CSV:  {ablation_outputs_dir / 'ablation_evaluation_comparison.csv'}")


if __name__ == "__main__":
    main()
