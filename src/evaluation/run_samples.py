import os
import sys
import time
import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------
# Project path setup
# Assumption: this file is placed in src/evaluation/run_evaluation.py
# Project root is two levels above: src/evaluation -> src -> project root
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.agent_builder import build_ktc_react_agent
from src.tools.document_tools import get_vector_db


# ---------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
MAIN_JSON_PATH = os.path.join(LOG_DIR, "generated_answers_structured.json")
RUN_METADATA_PATH = os.path.join(LOG_DIR, "generated_answers_run_metadata.json")


# ---------------------------------------------------------------------
# Evaluation cases
# Keep query IDs aligned with the answer key / evaluator.
# ---------------------------------------------------------------------
TEST_CASES: List[Dict[str, Any]] = [
    # 1. Data Agent
    {
        "query_number": 1,
        "query_id": "D1",
        "agent_category": "Data Agent",
        "query": "Filter the companies in the Scoring sheet by Region = `Asia` and list their Company Names and Total Benchmark Scores.",
    },
    {
        "query_number": 2,
        "query_id": "D2",
        "agent_category": "Data Agent",
        "query": "From the Non-Scored Research sheet, extract all companies that disclose sourcing from `China` and list their Company Names and Market Caps.",
    },
    {
        "query_number": 3,
        "query_id": "D3",
        "agent_category": "Data Agent",
        "query": "Clean the Scoring sheet by removing any rows with missing or zero Total Benchmark Scores, then list the top 5 companies by Total Benchmark Score descending.",
    },
    {
        "query_number": 4,
        "query_id": "D4",
        "agent_category": "Data Agent",
        "query": "Extract from the Detailed Scoring & Research sheet the comment for Amazon.com Inc. on indicator 1.1 (Supplier Code of Conduct).",
    },

    # 2. Analysis Agent
    {
        "query_number": 5,
        "query_id": "A1",
        "agent_category": "Analysis Agent",
        "query": "Compute the average Total Benchmark Score across all companies in the Scoring sheet, and the standard deviation.",
    },
    {
        "query_number": 6,
        "query_id": "A2",
        "agent_category": "Analysis Agent",
        "query": "Calculate the correlation between Market Cap and Total Benchmark Score using the Scoring sheet data.",
    },
    {
        "query_number": 7,
        "query_id": "A3",
        "agent_category": "Analysis Agent",
        "query": "Perform k-means clustering (k=3) on companies based on Total Benchmark Score and Purchasing Practices score from Scoring sheet, and list clusters.",
    },
    {
        "query_number": 8,
        "query_id": "A4",
        "agent_category": "Analysis Agent",
        "query": "Compute the median score for each theme (e.g., Commitment & Governance) across North American companies.",
    },

    # 3. External Validation / Research Agent
    {
        "query_number": 9,
        "query_id": "R1",
        "agent_category": "Research Agent",
        "query": "Cross-verify the high-risk sourcing countries for Amazon from the PDF with current ILO statistics on forced labour prevalence in China and Malaysia.",
    },
    {
        "query_number": 10,
        "query_id": "R2",
        "agent_category": "Research Agent",
        "query": "Search for the latest ILO report on forced labour in the ICT sector and compare it to the average KTC Remedy score (7/100).",
    },
    {
        "query_number": 11,
        "query_id": "R3",
        "agent_category": "Research Agent",
        # NOTE: corrected wording avoids reinforcing the invalid 79.5 benchmark score.
        "query": "Validate Samsung's top rank in the KTC Total Benchmark dataset by searching for recent news on its supply chain practices.",
    },
    {
        "query_number": 12,
        "query_id": "R4",
        "agent_category": "Research Agent",
        "query": "Fetch external data on global average market cap for ICT semiconductors and compare to KTC dataset average.",
    },

    # 4. Prediction Agent
    {
        "query_number": 13,
        "query_id": "P1",
        "agent_category": "Prediction Agent",
        "query": "Based on historical ranks (e.g., Amazon 2022 rank 8, 2025 rank 10), predict Amazon's 2027 rank if it improves Remedy by 10 points.",
    },
    {
        "query_number": 14,
        "query_id": "P2",
        "agent_category": "Prediction Agent",
        "query": "Project the industry average Total Benchmark Score for 2027 if scores increase by 5% annually.",
    },
    {
        "query_number": 15,
        "query_id": "P3",
        "agent_category": "Prediction Agent",
        "query": "Model score improvement for Asian companies if they match North America's average Purchasing Practices score. (5.77)",
    },
    {
        "query_number": 16,
        "query_id": "P4",
        "agent_category": "Prediction Agent",
        "query": "Predict the impact on Apple's score if it addresses Uyghur forced labour allegations from data.",
    },

    # 5. Synthesis Agent
    {
        "query_number": 17,
        "query_id": "S1",
        "agent_category": "Synthesis Agent",
        "query": "Synthesise key insights on top 5 companies' strengths and weaknesses from Scoring sheet.",
    },
    {
        "query_number": 18,
        "query_id": "S2",
        "agent_category": "Synthesis Agent",
        "query": "Compile a report on regional differences in Remedy scores, with ethical note on data biases.",
    },
    {
        "query_number": 19,
        "query_id": "S3",
        "agent_category": "Synthesis Agent",
        "query": "Synthesise opportunities for improvement from Amazon PDF.",
    },
    {
        "query_number": 20,
        "query_id": "S4",
        "agent_category": "Synthesis Agent",
        "query": "Compile insights on correlations between Market Cap and scores, noting ethical implications.",
    },

    # 6. Ethics Agent
    {
        "query_number": 21,
        "query_id": "E1",
        "agent_category": "Ethics Agent",
        "query": "Evaluate potential biases in the Scoring sheet data, using principal-agent theory.",
    },
    {
        "query_number": 22,
        "query_id": "E2",
        "agent_category": "Ethics Agent",
        "query": "Assess if low Remedy scores (avg 7) could amplify ethical risks in supply chains.",
    },
    {
        "query_number": 23,
        "query_id": "E3",
        "agent_category": "Ethics Agent",
        "query": "Check for regional bias in Non-Scored Research (e.g., more \"Yes\" for UK MSA in NA/Europe).",
    },
    {
        "query_number": 24,
        "query_id": "E4",
        "agent_category": "Ethics Agent",
        "query": "Evaluate ethical risks in predicting improvements for low-scorers like BOE (0 score).",
    },

    # 7. Text Mining Agent
    {
        "query_number": 25,
        "query_id": "T1",
        "agent_category": "Text Mining Agent",
        "query": "Extract and summarise key phrases from Amazon PDF on `Opportunities for Improvement`.",
    },
    {
        "query_number": 26,
        "query_id": "T2",
        "agent_category": "Text Mining Agent",
        "query": "Perform sentiment analysis on the Detailed Scoring & Research comments for Samsung.",
    },
    {
        "query_number": 27,
        "query_id": "T3",
        "agent_category": "Text Mining Agent",
        "query": "Mine the Amazon PDF for mentions of `forced labour` and categorise themes.",
    },
    {
        "query_number": 28,
        "query_id": "T4",
        "agent_category": "Text Mining Agent",
        "query": "Analyse text from Non-Scored Research on UK MSA compliance and identify patterns.",
    },
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return None


def compact_text(text: Any, max_chars: int = 20000) -> str:
    """Keep logs readable while preserving enough context for evaluation."""
    text = "" if text is None else str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED_FOR_LOG]"


def extract_retrieved_contexts(intermediate_steps: List[Any]) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract structured contexts from LangChain intermediate_steps.

    Works best if document_tools.py / research_tools.py return JSON payloads with:
      - retrieved_contexts: list[str|dict]
      - display: text shown to the LLM

    Also includes a fallback for older tools that return plain text.
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

        observation_record: Dict[str, Any] = {
            "step": idx,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "parsed_json": isinstance(parsed, dict),
        }

        if isinstance(parsed, dict):
            observation_record["payload"] = parsed
            contexts = parsed.get("retrieved_contexts") or []

            for rank, ctx in enumerate(contexts, start=1):
                if isinstance(ctx, dict):
                    content = str(ctx.get("content", "")).strip()
                    if not content:
                        continue
                    retrieved_contexts.append(content)
                    record = dict(ctx)
                    record.setdefault("content", content)
                    record.setdefault("rank", rank)
                    record.setdefault("tool_name", tool_name)
                    retrieved_context_records.append(record)
                elif isinstance(ctx, str):
                    content = ctx.strip()
                    if not content:
                        continue
                    retrieved_contexts.append(content)
                    retrieved_context_records.append({
                        "content": content,
                        "rank": rank,
                        "tool_name": tool_name,
                    })
        else:
            observation_record["observation"] = observation_text

            # Fallback for old plain-text RAG/web tools.
            # This is weaker than structured contexts but prevents total context loss.
            if tool_name in {"company_pdf_rag", "duckduckgo_web_search"} and observation_text.strip():
                retrieved_contexts.append(observation_text.strip())
                retrieved_context_records.append({
                    "content": observation_text.strip(),
                    "rank": len(retrieved_context_records) + 1,
                    "tool_name": tool_name,
                    "source": "plain_text_tool_observation",
                })

        tool_observations.append(observation_record)

    # Deduplicate while preserving order.
    seen = set()
    dedup_contexts: List[str] = []
    for ctx in retrieved_contexts:
        key = sha256_text(ctx)
        if key not in seen:
            seen.add(key)
            dedup_contexts.append(ctx)

    return dedup_contexts, retrieved_context_records, tool_observations


def atomic_write_json(path: str, payload: Any) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def save_results(results: List[Dict[str, Any]], run_metadata: Dict[str, Any]) -> None:
    """
    Main evaluator-compatible file is a JSON list.
    Separate metadata file avoids breaking evaluators that expect a list.
    """
    atomic_write_json(MAIN_JSON_PATH, results)
    atomic_write_json(RUN_METADATA_PATH, run_metadata)


def build_record(case: Dict[str, Any], result: Dict[str, Any], status: str, error: str | None, elapsed: float) -> Dict[str, Any]:
    final_answer = result.get("output", "") if isinstance(result, dict) else ""
    intermediate_steps = result.get("intermediate_steps", []) if isinstance(result, dict) else []
    retrieved_contexts, retrieved_context_records, tool_observations = extract_retrieved_contexts(intermediate_steps)

    return {
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


def main() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)

    run_started_utc = utc_now_iso()
    run_metadata: Dict[str, Any] = {
        "run_started_utc": run_started_utc,
        "run_finished_utc": None,
        "script_path": os.path.abspath(__file__),
        "project_root": PROJECT_ROOT,
        "total_queries": len(TEST_CASES),
        "main_json_path": MAIN_JSON_PATH,
        "notes": [
            "generated_answers_structured.json is intentionally a JSON list for evaluator compatibility.",
            "retrieved_contexts are extracted from tool intermediate_steps when available.",
            "For true RAG faithfulness, document/web tools should return structured JSON with retrieved_contexts.",
        ],
    }

    print("🚀 Initializing Evaluation Pipeline...")
    print(f"📁 JSON log will be saved to: {MAIN_JSON_PATH}")

    print("⏳ Loading Vector DB into memory...")
    try:
        get_vector_db()
        print("✅ Vector DB loaded.")
    except Exception as e:
        print(f"⚠️ Warning: Vector DB failed to load eagerly: {e}")

    print("🤖 Building Multi-Agent System...")
    agent_executor = build_ktc_react_agent()

    results: List[Dict[str, Any]] = []
    save_results(results, run_metadata)

    total_queries = len(TEST_CASES)
    print(f"📄 Starting evaluation of {total_queries} queries.\n")

    for case in TEST_CASES:
        qn = case["query_number"]
        qid = case["query_id"]
        query = case["query"]

        print(f"Evaluating [{qn}/{total_queries}] {qid}: {query[:80]}...")

        if getattr(agent_executor, "memory", None):
            agent_executor.memory.clear()

        start_time = time.time()
        response: Dict[str, Any] = {}
        error = None
        status = "SUCCESS"

        try:
            response = agent_executor.invoke({"input": query})
        except Exception as exc:
            status = "FAILED"
            error = str(exc)
            response = {"output": f"ERROR DURING EXECUTION: {error}", "intermediate_steps": []}

        elapsed = time.time() - start_time
        record = build_record(case, response, status, error, elapsed)
        results.append(record)

        # Save after every query, so partial runs are not lost.
        save_results(results, run_metadata)

        print(f"  -> {status} in {elapsed:.2f}s; contexts={len(record['retrieved_contexts'])}")

        # Small delay to reduce API rate-limit risk.
        time.sleep(2)

    run_metadata["run_finished_utc"] = utc_now_iso()
    run_metadata["completed_queries"] = len(results)
    run_metadata["success_count"] = sum(1 for r in results if r.get("mas_status") == "SUCCESS")
    run_metadata["failed_count"] = sum(1 for r in results if r.get("mas_status") == "FAILED")

    save_results(results, run_metadata)

    print("\n✅ Evaluation complete.")
    print(f"✅ Main JSON log saved to: {MAIN_JSON_PATH}")
    print(f"✅ Run metadata saved to: {RUN_METADATA_PATH}")


if __name__ == "__main__":
    main()
