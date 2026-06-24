"""Run MAS ablation variants and optionally evaluate them.

Place this folder at:
  src/evaluation/ablation/

Typical run from project root:
  python src/evaluation/ablation/run_ablations.py \
    --variants full_mas,llm_only,mas_without_ethics,mas_without_rag \
    --answer-key src/evaluation/logs/structured_reference_answer_key.json \
    --evaluate

Typical run from src/evaluation:
  python ablation/run_ablations.py --evaluate
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# Project path setup
# src/evaluation/ablation -> src/evaluation -> src -> project root
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ablations.ablation_agents import VARIANTS, build_ablation_agent
from src.evaluation.ablations.ablation_cases import TEST_CASES

DEFAULT_OUT_DIR = SCRIPT_DIR / "results"
DEFAULT_ANSWER_KEY = EVALUATION_DIR / "logs" / "structured_reference_answer_key.json"
DEFAULT_EVALUATOR = EVALUATION_DIR / "run_evaluation_full.py"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="replace")).hexdigest()


def compact_text(text: Any, max_chars: int = 20000) -> str:
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED_FOR_LOG]"


def safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return None


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def extract_retrieved_contexts(intermediate_steps: List[Any]) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract contexts from LangChain intermediate_steps.

    Structured RAG tools should return JSON with `retrieved_contexts` and `display`.
    Plain text fallback is retained for older tools.
    """
    retrieved_contexts: List[str] = []
    retrieved_context_records: List[Dict[str, Any]] = []
    tool_observations: List[Dict[str, Any]] = []

    for idx, step in enumerate(intermediate_steps or [], start=1):
        if isinstance(step, (tuple, list)) and len(step) == 2:
            action, observation = step
        else:
            continue

        tool_name = getattr(action, "tool", None) or getattr(action, "name", None) or "unknown_tool"
        tool_input = getattr(action, "tool_input", None)
        observation_text = compact_text(observation)
        parsed = safe_json_loads(observation)

        record: Dict[str, Any] = {
            "step": idx,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "parsed_json": isinstance(parsed, dict),
        }

        if isinstance(parsed, dict):
            record["payload"] = parsed
            contexts = parsed.get("retrieved_contexts") or []
            for rank, ctx in enumerate(contexts, start=1):
                if isinstance(ctx, dict):
                    content = str(ctx.get("content", "")).strip()
                    if not content:
                        continue
                    retrieved_contexts.append(content)
                    ctx_record = dict(ctx)
                    ctx_record.setdefault("content", content)
                    ctx_record.setdefault("rank", rank)
                    ctx_record.setdefault("tool_name", tool_name)
                    retrieved_context_records.append(ctx_record)
                elif isinstance(ctx, str):
                    content = ctx.strip()
                    if not content:
                        continue
                    retrieved_contexts.append(content)
                    retrieved_context_records.append({"content": content, "rank": rank, "tool_name": tool_name})
        else:
            record["observation"] = observation_text
            if tool_name in {"company_pdf_rag", "duckduckgo_web_search"} and observation_text.strip():
                retrieved_contexts.append(observation_text.strip())
                retrieved_context_records.append({
                    "content": observation_text.strip(),
                    "rank": len(retrieved_context_records) + 1,
                    "tool_name": tool_name,
                    "source": "plain_text_tool_observation",
                })

        tool_observations.append(record)

    seen = set()
    dedup_contexts: List[str] = []
    for ctx in retrieved_contexts:
        key = sha256_text(ctx)
        if key not in seen:
            seen.add(key)
            dedup_contexts.append(ctx)

    return dedup_contexts, retrieved_context_records, tool_observations


def build_record(case: Dict[str, Any], result: Dict[str, Any], status: str, error: Optional[str], elapsed: float, variant: str) -> Dict[str, Any]:
    final_answer = result.get("output", "") if isinstance(result, dict) else ""
    intermediate_steps = result.get("intermediate_steps", []) if isinstance(result, dict) else []
    retrieved_contexts, retrieved_context_records, tool_observations = extract_retrieved_contexts(intermediate_steps)

    return {
        "variant": variant,
        "query_number": case["query_number"],
        "query_id": case["query_id"],
        "agent_category": case["agent_category"],
        "query": case["query"],
        "final_answer": final_answer,
        "mas_status": status,
        "error": error,
        "latency_seconds": round(elapsed, 3),
        "run_timestamp_utc": utc_now_iso(),
        "retrieved_contexts": retrieved_contexts,
        "retrieved_context_records": retrieved_context_records,
        "tool_observations": tool_observations,
        "query_sha256": sha256_text(case["query"]),
        "final_answer_sha256": sha256_text(final_answer),
    }


def parse_variants(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(VARIANTS.keys())
    variants = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise ValueError(f"Unknown variant(s): {bad}. Valid variants: {sorted(VARIANTS)}")
    return variants


def select_cases(limit: Optional[int], only_ids: Optional[str]) -> List[Dict[str, Any]]:
    cases = TEST_CASES
    if only_ids:
        wanted = {x.strip() for x in only_ids.split(",") if x.strip()}
        cases = [c for c in cases if c["query_id"] in wanted]
    if limit is not None:
        cases = cases[:limit]
    return list(cases)


def run_variant(variant: str, cases: List[Dict[str, Any]], run_dir: Path, sleep_seconds: float = 0.0, resume: bool = False) -> Path:
    variant_dir = run_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    json_path = variant_dir / "generated_answers_structured.json"
    metadata_path = variant_dir / "generated_answers_run_metadata.json"

    if resume and json_path.exists():
        try:
            results: List[Dict[str, Any]] = json.loads(json_path.read_text(encoding="utf-8"))
            done_ids = {r.get("query_id") for r in results}
        except Exception:
            results = []
            done_ids = set()
    else:
        results = []
        done_ids = set()

    metadata = {
        "variant": variant,
        "variant_config": VARIANTS[variant].__dict__,
        "run_started_utc": utc_now_iso(),
        "run_finished_utc": None,
        "total_queries": len(cases),
        "project_root": str(PROJECT_ROOT),
        "script_path": str(Path(__file__).resolve()),
        "json_path": str(json_path),
        "notes": [
            "generated_answers_structured.json is a JSON list for evaluator compatibility.",
            "retrieved_contexts are extracted from tool intermediate_steps when available.",
        ],
    }
    atomic_write_json(metadata_path, metadata)
    atomic_write_json(json_path, results)

    print(f"\n=== Building variant: {variant} ===")
    print(VARIANTS[variant].description)
    agent_executor = build_ablation_agent(variant)

    for idx, case in enumerate(cases, start=1):
        qid = case["query_id"]
        if qid in done_ids:
            print(f"[{idx}/{len(cases)}] {variant} {qid}: already done, skipping")
            continue

        print(f"[{idx}/{len(cases)}] {variant} {qid}: {case['query'][:90]}...")
        if getattr(agent_executor, "memory", None):
            try:
                agent_executor.memory.clear()
            except Exception:
                pass

        start = time.time()
        response: Dict[str, Any] = {}
        status = "SUCCESS"
        error = None
        try:
            response = agent_executor.invoke({"input": case["query"]})
        except Exception as exc:
            status = "FAILED"
            error = str(exc)
            response = {"output": f"ERROR DURING EXECUTION: {error}", "intermediate_steps": []}

        elapsed = time.time() - start
        results.append(build_record(case, response, status, error, elapsed, variant))
        atomic_write_json(json_path, results)
        print(f"  -> {status} in {elapsed:.2f}s; contexts={len(results[-1]['retrieved_contexts'])}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    metadata["run_finished_utc"] = utc_now_iso()
    metadata["completed_queries"] = len(results)
    metadata["failed_queries"] = sum(1 for r in results if r.get("mas_status") != "SUCCESS")
    atomic_write_json(metadata_path, metadata)
    return json_path


def find_summary_file(eval_dir: Path) -> Optional[Path]:
    candidates = [
        eval_dir / "evaluation_summary.json",
        eval_dir / "ragas_style_summary.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def run_evaluator(generated_json: Path, answer_key: Path, eval_dir: Path, evaluator: Path) -> Optional[Path]:
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(evaluator),
        "--generated", str(generated_json),
        "--answer-key", str(answer_key),
        "--out-dir", str(eval_dir),
    ]
    print("\nRunning evaluator:")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(EVALUATION_DIR), text=True, capture_output=True)
    (eval_dir / "evaluator_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (eval_dir / "evaluator_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        print(f"Evaluator failed for {generated_json}")
        return None
    summary = find_summary_file(eval_dir)
    if summary:
        print(f"Evaluator summary: {summary}")
    else:
        print(f"Evaluator finished but no known summary file found in {eval_dir}")
    return summary


def extract_summary_metrics(summary_path: Path, variant: str) -> Dict[str, Any]:
    obj = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = obj.get("summary", obj)
    # Support both old/new section names.
    all_rows = summary.get("all_rows") or summary.get("all_28_rows") or summary.get("final_accuracy_rows_only") or {}
    final_rows = summary.get("final_accuracy_rows_only") or all_rows
    rag_rows = summary.get("rows_with_retrieved_contexts") or summary.get("rag_rows_with_retrieved_contexts") or {}
    return {
        "variant": variant,
        "summary_path": str(summary_path),
        "n": all_rows.get("n"),
        "pass": all_rows.get("pass"),
        "partial": all_rows.get("partial"),
        "fail": all_rows.get("fail"),
        "strict_pass_rate": all_rows.get("strict_pass_rate"),
        "pass_or_partial_rate": all_rows.get("pass_or_partial_rate"),
        "mean_answer_correctness": all_rows.get("mean_answer_correctness"),
        "mean_answer_relevancy": all_rows.get("mean_answer_relevancy"),
        "rows_with_contexts": all_rows.get("rows_with_contexts"),
        "mean_rag_composite": all_rows.get("mean_rag_composite"),
        "mean_faithfulness": all_rows.get("mean_faithfulness"),
        "mean_context_precision": all_rows.get("mean_context_precision"),
        "mean_context_recall": all_rows.get("mean_context_recall"),
        "final_strict_pass_rate": final_rows.get("strict_pass_rate"),
        "rag_n": rag_rows.get("n"),
        "rag_strict_pass_rate": rag_rows.get("strict_pass_rate"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MAS ablation variants and optionally evaluate outputs.")
    parser.add_argument("--variants", default="full_mas,llm_only,mas_without_ethics,mas_without_rag", help="Comma-separated variants or 'all'.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for ablation outputs.")
    parser.add_argument("--run-id", default=None, help="Optional run ID. Defaults to UTC timestamp.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries for a smoke test.")
    parser.add_argument("--only-ids", default=None, help="Comma-separated query IDs, e.g. R1,R2,E1.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between queries.")
    parser.add_argument("--resume", action="store_true", help="Skip already completed query IDs in existing JSON logs.")
    parser.add_argument("--evaluate", action="store_true", help="After generating outputs, run the evaluator for each variant.")
    parser.add_argument("--answer-key", default=str(DEFAULT_ANSWER_KEY), help="Structured answer key for evaluation.")
    parser.add_argument("--evaluator", default=str(DEFAULT_EVALUATOR), help="Path to run_evaluation_full.py direct evaluator.")
    args = parser.parse_args()

    variants = parse_variants(args.variants)
    cases = select_cases(args.limit, args.only_ids)
    run_id = args.run_id or timestamp_id()
    run_dir = Path(args.out_dir).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    index_rows: List[Dict[str, Any]] = []
    aggregate_metrics: List[Dict[str, Any]] = []

    for variant in variants:
        generated_json = run_variant(
            variant=variant,
            cases=cases,
            run_dir=run_dir,
            sleep_seconds=args.sleep,
            resume=args.resume,
        )
        row = {
            "variant": variant,
            "generated_json": str(generated_json),
            "description": VARIANTS[variant].description,
        }

        if args.evaluate:
            answer_key = Path(args.answer_key).expanduser().resolve()
            evaluator = Path(args.evaluator).expanduser().resolve()
            eval_dir = generated_json.parent / "evaluation"
            summary_path = run_evaluator(generated_json, answer_key, eval_dir, evaluator)
            row["evaluation_summary"] = str(summary_path) if summary_path else ""
            if summary_path:
                aggregate_metrics.append(extract_summary_metrics(summary_path, variant))

        index_rows.append(row)

    atomic_write_json(run_dir / "ablation_outputs_index.json", index_rows)
    write_csv(run_dir / "ablation_outputs_index.csv", index_rows)
    if aggregate_metrics:
        atomic_write_json(run_dir / "ablation_comparison_summary.json", aggregate_metrics)
        write_csv(run_dir / "ablation_comparison_summary.csv", aggregate_metrics)

    print("\nAblation run complete.")
    print(f"Run directory: {run_dir}")
    print(f"Index: {run_dir / 'ablation_outputs_index.csv'}")
    if aggregate_metrics:
        print(f"Comparison: {run_dir / 'ablation_comparison_summary.csv'}")


if __name__ == "__main__":
    main()
