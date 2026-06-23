"""
Corrected final MAS evaluator (no command-line parameters).

Use with:
  logs/structured_answer_key_MAS_evaluation_final.json
  logs/generated_answers_structured.json

Run:
  python evaluate_mas_outputs_FINAL_CORRECTED.py

Outputs:
  outputs/mas_evaluation_results_final_corrected.json
  outputs/mas_evaluation_summary_final_corrected.csv
  outputs/mas_evaluation_category_summary_final_corrected.csv
  outputs/mas_evaluation_rag_metrics_final_corrected.csv

This corrected version fixes the main scoring problems found in the previous evaluator:
- Does not treat score text after "— 61.00" as part of the company name.
- Uses line-level entity-value matching before sentence splitting, so "Co. Ltd." does not break matches.
- Separates HP Inc. from HPE / Hewlett Packard Enterprise.
- Does not penalize top-k ranking answers because of rank numbers such as "1." or "top 5".
- Accepts rounded approximate values such as "about 7" for 6.98 where the query/reference uses approximate wording.
- Keeps NEEDS_CORRECTION / NEEDS_SOURCE_SNAPSHOT / NEEDS_VALIDATION rows out of final accuracy.
- Does not claim Context Precision / Context Recall when retrieved_contexts are missing.
- Uses stricter thresholds: PASS >= 0.85, PARTIAL >= 0.60, FAIL < 0.60.

BERTScore is intentionally not included.
RAGAS library is intentionally not required; RAG-style metrics are transparent heuristics.
"""

from __future__ import annotations

import csv
import json
import os
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ============================================================
# HARD-CODED SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ANSWER_KEY_CANDIDATES = [
    BASE_DIR / "logs" / "structured_answer_key_MAS_evaluation_final.json",
    BASE_DIR / "structured_answer_key_MAS_evaluation_final.json",
    Path("/mnt/data/structured_answer_key_MAS_evaluation_final.json"),
    # Sandbox fallback for the file uploaded in this conversation.
    Path("/mnt/data/d687c9c7-c98a-4d2a-a902-399d7080e070.json"),
]

GENERATED_ANSWERS_CANDIDATES = [
    BASE_DIR / "logs" / "generated_answers_structured.json",
    BASE_DIR / "generated_answers_structured.json",
    BASE_DIR / "generated_answers_structured_exact.json",
    Path("/mnt/data/generated_answers_structured.json"),
    Path("/mnt/data/generated_answers_structured_exact.json"),
    # Sandbox fallback for the file uploaded in this conversation.
    Path("/mnt/data/1dd06bb5-32fb-4c77-9e11-3cd04f4a7bd9.json"),
]

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_JSON_PATH = OUTPUT_DIR / "mas_evaluation_results_final_corrected.json"
OUTPUT_CSV_PATH = OUTPUT_DIR / "mas_evaluation_summary_final_corrected.csv"
OUTPUT_CATEGORY_CSV_PATH = OUTPUT_DIR / "mas_evaluation_category_summary_final_corrected.csv"
OUTPUT_RAG_CSV_PATH = OUTPUT_DIR / "mas_evaluation_rag_metrics_final_corrected.csv"

USE_LLM_JUDGE = False
JUDGE_MODEL = "gpt-4.1"

QUERY_ID_ORDER = [
    "D1", "D2", "D3", "D4",
    "A1", "A2", "A3", "A4",
    "R1", "R2", "R3", "R4",
    "P1", "P2", "P3", "P4",
    "S1", "S2", "S3", "S4",
    "E1", "E2", "E3", "E4",
    "T1", "T2", "T3", "T4",
]

FINAL_KEY_STATUSES = {"CLEAN", "CLEAN_WITH_RUBRIC", "CLEAN_WITH_METHOD_NOTE"}
DIAGNOSTIC_ONLY_STATUSES = {"NEEDS_CORRECTION", "NEEDS_SOURCE_SNAPSHOT", "NEEDS_VALIDATION"}

OPEN_ENDED_TYPES = {
    "rag", "synthesis", "ethics", "external", "semantic", "claim", "qualitative", "text", "document",
}
RAG_RELATED_TYPES = {"rag", "document", "pdf", "text mining", "retrieval", "external"}

PASS_THRESHOLD = 0.85
PARTIAL_THRESHOLD = 0.60
CLAIM_KEYWORD_COVERAGE_THRESHOLD = 0.45
CONTEXT_RELEVANCE_THRESHOLD = 0.12
FAITHFULNESS_SENTENCE_SUPPORT_THRESHOLD = 0.18
INCIDENTAL_NUMBERS = {2022, 2023, 2024, 2025, 2026, 2027, 100}


# ============================================================
# IO HELPERS
# ============================================================

def first_existing_path(candidates: List[Path], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{label} not found. Checked:\n{checked}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def safe_json_loads(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower().replace("labour", "labor")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9.\s&()/%+\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_entity(text: Any) -> str:
    text = normalize_text(text)
    text = text.replace(".", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def content_tokens(text: Any) -> List[str]:
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "by", "as", "is", "are", "be", "was", "were", "this", "that", "these",
        "those", "from", "into", "not", "but", "if", "then", "there", "where",
        "which", "who", "what", "when", "how", "does", "do", "did", "has",
        "have", "had", "can", "could", "should", "would", "may", "might",
        "must", "include", "includes", "including", "required", "claim",
        "answer", "response", "score", "scores", "query", "question", "company",
        "companies", "data", "sheet", "table", "using", "based", "overall",
        "also", "more", "less", "than", "about", "across", "such", "e.g",
        "approximately", "approx", "average", "current",
    }
    return [token for token in tokenize(text) if len(token) > 2 and token not in stopwords]


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(items))


def keyword_set(text: Any) -> set:
    return set(content_tokens(text))


def lexical_overlap_score(source: str, target: str) -> float:
    source_keywords = keyword_set(source)
    target_keywords = keyword_set(target)
    if not source_keywords:
        return 0.0
    return len(source_keywords & target_keywords) / len(source_keywords)


def f1_from_precision_recall(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def strip_list_prefix(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[-*•]\s*", "", line)
    line = re.sub(r"^\d+\.\s*", "", line)
    return line.strip()


def answer_lines(text: str) -> List[str]:
    return [strip_list_prefix(line) for line in str(text).splitlines() if strip_list_prefix(line)]


def split_sentences(text: str) -> List[str]:
    # Sentence fallback only. Do not use this before line matching because company names such as "Co. Ltd."
    # contain periods that can break sentence splitting.
    text = str(text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+|(?:\s*[-•]\s+)", text)
    return [part.strip() for part in parts if len(part.strip()) > 20]


# ============================================================
# NUMBER PARSING
# ============================================================

def extract_numbers(text: Any) -> List[float]:
    return [float(x) for x in re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", str(text))]


def extract_numbers_without_list_ranks(text: Any) -> List[float]:
    numbers: List[float] = []
    for line in str(text).splitlines():
        clean = strip_list_prefix(line)
        numbers.extend(extract_numbers(clean))
    if not numbers and str(text).strip():
        numbers.extend(extract_numbers(text))
    return numbers


def filter_reference_numbers(numbers: List[float]) -> List[float]:
    return [number for number in numbers if int(number) not in INCIDENTAL_NUMBERS]


def has_approximate_language(text: str) -> bool:
    return bool(re.search(r"\b(about|approx|approximately|around|roughly|≈)\b", str(text), re.I))


def effective_tolerance(target: float, base_tolerance: float, answer: str, reference: str = "") -> float:
    # Keep exact tolerance by default. If the answer/reference explicitly uses approximate wording,
    # allow common rounding such as 6.98 -> about 7.
    if has_approximate_language(answer) or has_approximate_language(reference):
        if abs(target) >= 1:
            return max(base_tolerance, 0.05)
    return base_tolerance


def number_present(target: float, numbers: List[float], tolerance: float = 0.01, answer: str = "",
                   reference: str = "") -> bool:
    tol = effective_tolerance(target, tolerance, answer, reference)
    return any(abs(number - target) <= tol for number in numbers)


def parse_tolerance(value: Optional[str], default: float = 0.01) -> float:
    if not value:
        return default
    match = re.search(r"±\s*([0-9.]+)", str(value))
    return float(match.group(1)) if match else default


# ============================================================
# ENTITY PARSING
# ============================================================

LEGAL_SUFFIX_PATTERN = re.compile(
    r"\b(co|corp|corporation|inc|incorporated|ltd|plc|ag|oyj|sa|s\.a|nv|company|technologies|technology|electronics|group|holdings)\b\.?",
    re.I,
)

COMPANY_MARKERS = [
    "Inc", "Corp", "Co.", "Ltd", "PLC", "AG", "Oyj", "S.A.", "NV",
    "Samsung", "Apple", "Cisco", "BOE", "HPE", "Xiaomi", "Panasonic",
    "Canon", "Sony", "NVIDIA", "Amazon", "Qualcomm", "Seagate", "NXP",
    "Logitech", "Dell", "HP", "Fujifilm", "Kyocera", "Keyence", "Murata",
    "Luxshare", "Infineon", "Ericsson", "AMD", "Advanced Micro Devices",
    "Hewlett Packard", "Hon Hai", "Foxconn", "Taiwan Semiconductor",
    "Semiconductor Manufacturing", "Best Buy", "Corning", "Broadcom", "Amphenol", "Nokia",
    "SK Hynix", "LG Electronics",
]


def split_reference_items(reference_answer: str) -> List[str]:
    items = []
    for line in str(reference_answer).splitlines():
        clean = strip_list_prefix(line)
        if clean:
            items.append(clean)
    return items


def remove_score_suffix_from_name(text: str) -> str:
    text = str(text).strip()
    # Common reference formats:
    # Company: 61.00
    # Company — 61.00
    # Company - 61.00
    # Company with a score of 61.00
    # Company: Market Cap = 235.74B
    text = re.split(r"\s+with a score\b", text, maxsplit=1, flags=re.I)[0]
    text = re.split(r"\bmarket cap\b", text, maxsplit=1, flags=re.I)[0]
    text = re.split(r"\s+[—–-]\s+(?=\d)", text, maxsplit=1)[0]
    if ":" in text:
        before, after = text.split(":", 1)
        if re.search(r"\d|market cap", after, re.I):
            text = before
    text = re.sub(r"\s*=\s*[-+]?\d.*$", "", text)
    return text.strip(" :=-—–")


def company_name_from_item(item: str) -> str:
    return remove_score_suffix_from_name(strip_list_prefix(item))


def looks_like_company_name(name: str) -> bool:
    name = str(name).strip()
    if not name:
        return False

    # Legal suffixes must be standalone tokens; otherwise "AG" would match "Average".
    legal_suffixes = ["Inc", "Corp", "Co.", "Ltd", "PLC", "AG", "Oyj", "S.A.", "NV"]
    for suffix in legal_suffixes:
        suffix_norm = re.escape(suffix.lower().replace(".", ""))
        name_norm = normalize_for_entity(name)
        if re.search(r"(?<![a-z0-9])" + suffix_norm + r"(?![a-z0-9])", name_norm):
            return True

    named_markers = [
        marker for marker in COMPANY_MARKERS
        if marker not in legal_suffixes and len(marker) > 2
    ]
    name_norm = normalize_for_entity(name)
    for marker in named_markers:
        marker_norm = normalize_for_entity(marker)
        if re.search(r"(?<![a-z0-9])" + re.escape(marker_norm) + r"(?![a-z0-9])", name_norm):
            return True

    return False


def canonical_entity_key(name: str) -> str:
    text = normalize_for_entity(name)
    text = LEGAL_SUFFIX_PATTERN.sub(" ", text)
    text = re.sub(r"\b(the|publ)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def entity_aliases(name: str) -> List[str]:
    raw = normalize_for_entity(remove_score_suffix_from_name(name))
    aliases = [raw]

    # Special cases to prevent HP Inc. from matching HPE.
    if raw in {"hp inc", "hp"} or re.fullmatch(r"hp inc", raw):
        return ["hp inc"]

    if "hewlett packard enterprise" in raw or "(hpe)" in raw or raw == "hpe":
        aliases.extend(["hewlett packard enterprise", "hpe"])

    if "hon hai precision" in raw or "foxconn" in raw:
        aliases.extend(["hon hai precision", "foxconn"])

    if "advanced micro devices" in raw:
        aliases.extend(["advanced micro devices", "amd"])

    if "telefonaktiebolaget lm ericsson" in raw:
        aliases.extend(["telefonaktiebolaget lm ericsson", "ericsson"])

    if "nvidia" in raw:
        aliases.append("nvidia")

    if "amazon.com" in raw:
        aliases.extend(["amazon.com inc", "amazon com inc", "amazon"])

    if "samsung electronics" in raw:
        aliases.append("samsung electronics")

    if "apple inc" in raw:
        aliases.append("apple inc")

    key = canonical_entity_key(raw)
    if len(key) > 3:
        aliases.append(key)

    return unique_preserve_order([alias for alias in aliases if alias])


def contains_alias_as_phrase(text: str, alias: str) -> bool:
    text_norm = normalize_for_entity(text)
    alias_norm = normalize_for_entity(alias)

    if not alias_norm:
        return False

    # HP Inc. must match HP Inc., not HPE or Hewlett Packard Enterprise.
    if alias_norm == "hp inc":
        return bool(re.search(r"\bhp\s+inc\b", text_norm))

    # HPE should match acronym or full company only.
    if alias_norm == "hpe":
        return bool(re.search(r"\bhpe\b", text_norm))

    pattern = r"(?<![a-z0-9])" + re.escape(alias_norm) + r"(?![a-z0-9])"
    return bool(re.search(pattern, text_norm))


def name_present(name: str, answer: str) -> bool:
    return any(contains_alias_as_phrase(answer, alias) for alias in entity_aliases(name))


def extract_expected_entities(reference_answer: str) -> List[str]:
    entities = []
    for item in split_reference_items(reference_answer):
        name = company_name_from_item(item)
        if looks_like_company_name(name):
            entities.append(name)
    return unique_preserve_order(entities)


def build_global_entity_universe(answer_key: Dict[str, Any]) -> List[str]:
    entities = []
    for row in answer_key.get("answer_key", []):
        entities.extend(extract_expected_entities(row.get("clean_reference_answer", "")))
    return unique_preserve_order(entities)


def extract_predicted_entities(answer: str, global_entities: List[str]) -> List[str]:
    return [entity for entity in global_entities if name_present(entity, answer)]


def extract_expected_entity_value_pairs(reference_answer: str) -> List[Dict[str, Any]]:
    pairs = []
    for item in split_reference_items(reference_answer):
        entity = company_name_from_item(item)
        if not looks_like_company_name(entity):
            continue
        values = filter_reference_numbers(extract_numbers_without_list_ranks(item))
        if values:
            pairs.append({"entity": entity, "values": values})
    return pairs


def line_or_sentence_with_entity(answer: str, entity: str) -> str:
    # Line-level matching first avoids breaking "Co. Ltd." into fake sentences.
    for line in answer_lines(answer):
        if name_present(entity, line):
            return line

    for sentence in split_sentences(answer):
        if name_present(entity, sentence):
            return sentence

    return ""


# ============================================================
# GENERATED ANSWER LOADING
# ============================================================

def parse_text_mas_log(log_text: str) -> Dict[str, Dict[str, Any]]:
    pattern = re.compile(
        r"--- Query\s+(\d+)/28\s+\[(.*?)\]\s+---\s*"
        r"Time Taken\s*:\s*([0-9.]+)\s*seconds\s*"
        r"Question\s*:\s*(.*?)\n"
        r"Answer\s*:\s*\n(.*?)(?=\n-+\n\n--- Query|\n-+\s*$)",
        re.S,
    )

    outputs: Dict[str, Dict[str, Any]] = {}
    for match in pattern.finditer(log_text):
        query_number = int(match.group(1))
        if not 1 <= query_number <= len(QUERY_ID_ORDER):
            continue

        query_id = QUERY_ID_ORDER[query_number - 1]
        outputs[query_id] = {
            "query_number": query_number,
            "query_id": query_id,
            "mas_status": match.group(2).strip(),
            "latency_seconds": float(match.group(3)),
            "question": match.group(4).strip(),
            "answer": match.group(5).strip(),
            "retrieved_contexts": [],
            "raw_record": None,
        }

    return outputs


def normalize_json_mas_record(record: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    query_number = record.get("query_number") or record.get("query_num") or record.get("number") or index + 1
    try:
        query_number = int(query_number)
    except Exception:
        query_number = index + 1

    query_id = record.get("query_id")
    if not query_id and 1 <= query_number <= len(QUERY_ID_ORDER):
        query_id = QUERY_ID_ORDER[query_number - 1]

    if not query_id:
        return None

    answer = (
            record.get("answer")
            or record.get("response")
            or record.get("final_answer")
            or record.get("generated_answer")
            or record.get("output")
            or ""
    )

    question = record.get("question") or record.get("query") or record.get("prompt") or ""

    contexts = (
            record.get("retrieved_contexts")
            or record.get("contexts")
            or record.get("context")
            or record.get("retrieved_chunks")
            or record.get("source_contexts")
            or []
    )
    if isinstance(contexts, str):
        contexts = [contexts]
    elif isinstance(contexts, list):
        contexts = [str(item) for item in contexts]
    else:
        contexts = []

    latency = (
            record.get("latency_seconds")
            or record.get("time_taken")
            or record.get("time_taken_seconds")
            or record.get("duration")
            or None
    )
    try:
        latency = float(latency) if latency is not None else None
    except Exception:
        latency = None

    status = str(record.get("mas_status") or record.get("status") or "UNKNOWN")

    return {
        "query_number": query_number,
        "query_id": query_id,
        "mas_status": status,
        "latency_seconds": latency,
        "question": str(question),
        "answer": str(answer),
        "retrieved_contexts": contexts,
        "raw_record": record,
    }


def load_mas_outputs(path: Path) -> Dict[str, Dict[str, Any]]:
    text = read_text(path)
    obj = safe_json_loads(text)

    if obj is not None:
        if isinstance(obj, dict):
            records = obj.get("results") or obj.get("queries") or obj.get("items") or obj.get("outputs") or []
            if isinstance(records, dict):
                records = list(records.values())
        elif isinstance(obj, list):
            records = obj
        else:
            records = []

        outputs = {}
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            normalized = normalize_json_mas_record(record, index)
            if normalized:
                outputs[normalized["query_id"]] = normalized
        if outputs:
            return outputs

    # JSONL fallback.
    outputs = {}
    for index, line in enumerate(text.splitlines()):
        record = safe_json_loads(line.strip())
        if isinstance(record, dict):
            normalized = normalize_json_mas_record(record, index)
            if normalized:
                outputs[normalized["query_id"]] = normalized
    if outputs:
        return outputs

    return parse_text_mas_log(text)


# ============================================================
# CLAIM-LEVEL EVALUATION
# ============================================================

def build_claim_index(answer_key: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    claims_by_qid: Dict[str, List[Dict[str, Any]]] = {}
    for claim in answer_key.get("atomic_claims", []):
        query_id = claim.get("query_id")
        if query_id:
            claims_by_qid.setdefault(query_id, []).append(claim)
    return claims_by_qid


def claim_keywords(claim_text: str) -> List[str]:
    return unique_preserve_order(content_tokens(claim_text))


def score_atomic_claims(
        query_id: str,
        answer: str,
        claims_by_qid: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    claims = claims_by_qid.get(query_id, [])
    if not claims:
        return None

    normalized_answer = normalize_text(answer)
    claim_results = []
    earned_weight = 0.0
    total_weight = 0.0

    for claim in claims:
        required_claim = claim.get("required_claim", "")
        weight = float(claim.get("weight", 1) or 1)
        keywords = claim_keywords(required_claim)

        if not keywords:
            keyword_overlap = 0.0
            covered = False
        else:
            hits = sum(1 for kw in keywords if kw in normalized_answer)
            keyword_overlap = hits / len(keywords)
            covered = keyword_overlap >= CLAIM_KEYWORD_COVERAGE_THRESHOLD

        # Numeric claim guardrails.
        normalized_claim = normalize_text(required_claim)
        for exact_term in ["0.323", "6.98", "3.31", "9.43", "8.86", "20.22", "14.23", "0.061", "0.339"]:
            if exact_term in normalized_claim and exact_term not in normalized_answer:
                # Accept 6.98 when the answer says about 7.
                if exact_term == "6.98" and re.search(r"\b(about|approximately|approx|around)\s+7\b",
                                                      normalized_answer):
                    pass
                else:
                    covered = False

        total_weight += weight
        if covered:
            earned_weight += weight

        claim_results.append(
            {
                "claim_id": claim.get("claim_id"),
                "required_claim": required_claim,
                "weight": weight,
                "covered_auto": covered,
                "keyword_overlap": round(keyword_overlap, 3),
            }
        )

    score = earned_weight / total_weight if total_weight else None

    return {
        "claim_score": round(score, 3) if score is not None else None,
        "completeness": round(score, 3) if score is not None else None,
        "covered_claims": sum(1 for item in claim_results if item["covered_auto"]),
        "total_claims": len(claim_results),
        "claim_results": claim_results,
    }


# ============================================================
# DETERMINISTIC EVALUATION
# ============================================================

def numeric_metrics(reference: str, answer: str, tolerance: float) -> Dict[str, Any]:
    reference_numbers = filter_reference_numbers(extract_numbers_without_list_ranks(reference))
    answer_numbers = filter_reference_numbers(extract_numbers_without_list_ranks(answer))

    found = [
        number for number in reference_numbers
        if number_present(number, answer_numbers, tolerance, answer=answer, reference=reference)
    ]
    missing = [
        number for number in reference_numbers
        if not number_present(number, answer_numbers, tolerance, answer=answer, reference=reference)
    ]

    matched_answer_indices = set()
    for ref_number in reference_numbers:
        tol = effective_tolerance(ref_number, tolerance, answer, reference)
        for index, answer_number in enumerate(answer_numbers):
            if index not in matched_answer_indices and abs(answer_number - ref_number) <= tol:
                matched_answer_indices.add(index)
                break

    extra = [number for index, number in enumerate(answer_numbers) if index not in matched_answer_indices]

    recall = len(found) / len(reference_numbers) if reference_numbers else None

    # Precision is useful diagnostically, but final scoring relies mainly on recall/pairs/ranking because
    # generated text often contains harmless numbers in introductions such as "top 5".
    precision = len(matched_answer_indices) / len(answer_numbers) if answer_numbers else None
    f1 = f1_from_precision_recall(precision, recall)

    nearest_errors = []
    for ref_number in reference_numbers:
        if answer_numbers:
            nearest_errors.append(min(abs(ans_number - ref_number) for ans_number in answer_numbers))

    return {
        "reference_numbers": reference_numbers,
        "answer_numbers": answer_numbers,
        "numeric_found": found,
        "numeric_missing": missing,
        "numeric_extra": extra,
        "numeric_precision": round(precision, 3) if precision is not None else None,
        "numeric_recall": round(recall, 3) if recall is not None else None,
        "numeric_f1": round(f1, 3) if f1 is not None else None,
        "mean_absolute_error_nearest": round(statistics.mean(nearest_errors), 4) if nearest_errors else None,
        "numeric_tolerance": tolerance,
    }


def entity_metrics(expected_entities: List[str], predicted_entities: List[str]) -> Dict[str, Any]:
    expected = set(expected_entities)
    predicted = set(predicted_entities)

    true_positive = expected & predicted
    missing = expected - predicted
    extra = predicted - expected

    precision = len(true_positive) / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = len(true_positive) / len(expected) if expected else None
    f1 = f1_from_precision_recall(precision, recall) if recall is not None else None

    return {
        "expected_entities": sorted(expected),
        "predicted_entities": sorted(predicted),
        "entity_true_positive": sorted(true_positive),
        "entity_missing": sorted(missing),
        "entity_extra": sorted(extra),
        "entity_precision": round(precision, 3) if precision is not None else None,
        "entity_recall": round(recall, 3) if recall is not None else None,
        "entity_f1": round(f1, 3) if f1 is not None else None,
    }


def entity_value_pair_metrics(reference: str, answer: str, tolerance: float) -> Dict[str, Any]:
    pairs = extract_expected_entity_value_pairs(reference)
    if not pairs:
        return {
            "expected_pairs": [],
            "correct_pairs": [],
            "incorrect_or_missing_pairs": [],
            "entity_value_pair_accuracy": None,
        }

    correct = []
    incorrect_or_missing = []

    for pair in pairs:
        entity = pair["entity"]
        expected_values = pair["values"]
        matched_text = line_or_sentence_with_entity(answer, entity)
        matched_numbers = extract_numbers_without_list_ranks(matched_text)

        value_ok = all(
            number_present(value, matched_numbers, tolerance, answer=matched_text, reference=reference)
            for value in expected_values
        )
        entity_ok = bool(matched_text)

        if entity_ok and value_ok:
            correct.append(pair)
        else:
            incorrect_or_missing.append(
                {
                    "entity": entity,
                    "expected_values": expected_values,
                    "matched_text": matched_text,
                    "matched_numbers": matched_numbers,
                }
            )

    accuracy = len(correct) / len(pairs) if pairs else None

    return {
        "expected_pairs": pairs,
        "correct_pairs": correct,
        "incorrect_or_missing_pairs": incorrect_or_missing,
        "entity_value_pair_accuracy": round(accuracy, 3) if accuracy is not None else None,
    }


def ranking_metrics(reference: str, answer: str, expected_entities: List[str]) -> Dict[str, Any]:
    if not expected_entities:
        return {
            "ordered_topk_accuracy": None,
            "topk_membership_accuracy": None,
            "predicted_order": [],
            "expected_order": [],
        }

    expected_order = expected_entities
    found_with_position = []

    for entity in expected_order:
        best_pos = None
        for alias in entity_aliases(entity):
            answer_norm = normalize_for_entity(answer)
            alias_norm = normalize_for_entity(alias)
            match = re.search(r"(?<![a-z0-9])" + re.escape(alias_norm) + r"(?![a-z0-9])", answer_norm)
            if match:
                if best_pos is None or match.start() < best_pos:
                    best_pos = match.start()
        if best_pos is not None:
            found_with_position.append((best_pos, entity))

    predicted_order = [entity for _, entity in sorted(found_with_position)]

    membership_accuracy = len(set(predicted_order) & set(expected_order)) / len(expected_order)
    same_position = sum(
        1 for index, entity in enumerate(expected_order)
        if index < len(predicted_order) and predicted_order[index] == entity
    )
    ordered_accuracy = same_position / len(expected_order)

    return {
        "expected_order": expected_order,
        "predicted_order": predicted_order,
        "topk_membership_accuracy": round(membership_accuracy, 3),
        "ordered_topk_accuracy": round(ordered_accuracy, 3),
    }


def unit_consistency(reference: str, answer: str) -> Dict[str, Any]:
    ref_market_billion = "Market Cap" in str(reference) and re.search(r"\bB\b|billion", str(reference), re.I)
    ans_million = re.search(r"\bmillion\b", str(answer), re.I)
    issue = bool(ref_market_billion and ans_million)
    return {
        "unit_issue": issue,
        "unit_issue_type": "market_cap_million_vs_billion" if issue else None,
    }


def deterministic_structured_eval(
        row: Dict[str, Any],
        answer: str,
        global_entities: List[str],
) -> Tuple[Dict[str, Any], Optional[float]]:
    reference = row.get("clean_reference_answer", "")
    tolerance = parse_tolerance(row.get("numeric_tolerance"), default=0.01)
    evaluation_type = row.get("evaluation_type", "").lower()

    expected_entities = extract_expected_entities(reference)
    predicted_entities = extract_predicted_entities(answer, global_entities)

    n_metrics = numeric_metrics(reference, answer, tolerance)
    e_metrics = entity_metrics(expected_entities, predicted_entities)
    pair_metrics = entity_value_pair_metrics(reference, answer, tolerance)
    rank_metrics = ranking_metrics(reference, answer, expected_entities)
    unit_metrics = unit_consistency(reference, answer)

    available_scores: List[float] = []

    if "top-k" in evaluation_type or "ranking" in evaluation_type:
        if rank_metrics["ordered_topk_accuracy"] is not None:
            available_scores.append(rank_metrics["ordered_topk_accuracy"])
        if pair_metrics["entity_value_pair_accuracy"] is not None:
            available_scores.append(pair_metrics["entity_value_pair_accuracy"])
        if n_metrics["numeric_recall"] is not None and n_metrics["reference_numbers"]:
            available_scores.append(n_metrics["numeric_recall"])

    elif "list" in evaluation_type or "entity" in evaluation_type or "exact" in evaluation_type:
        if e_metrics["entity_f1"] is not None and expected_entities:
            available_scores.append(e_metrics["entity_f1"])
        if pair_metrics["entity_value_pair_accuracy"] is not None:
            available_scores.append(pair_metrics["entity_value_pair_accuracy"])
        if n_metrics["numeric_recall"] is not None and n_metrics["reference_numbers"]:
            available_scores.append(n_metrics["numeric_recall"])

    else:
        if n_metrics["numeric_recall"] is not None and n_metrics["reference_numbers"]:
            available_scores.append(n_metrics["numeric_recall"])
        if e_metrics["entity_f1"] is not None and expected_entities:
            available_scores.append(e_metrics["entity_f1"])
        if pair_metrics["entity_value_pair_accuracy"] is not None:
            available_scores.append(pair_metrics["entity_value_pair_accuracy"])

    if unit_metrics["unit_issue"]:
        available_scores.append(0.85)

    score = min(available_scores) if available_scores else None

    details = {
        "numeric_metrics": n_metrics,
        "entity_metrics": e_metrics,
        "entity_value_pair_metrics": pair_metrics,
        "ranking_metrics": rank_metrics,
        "unit_consistency": unit_metrics,
    }

    return details, round(score, 3) if score is not None else None


# ============================================================
# RAG / DOCUMENT-GROUNDED EVALUATION
# ============================================================

def is_rag_related(row: Dict[str, Any]) -> bool:
    text = f"{row.get('evaluation_type', '')} {row.get('agent_category', '')} {row.get('query_id', '')}".lower()
    return any(marker in text for marker in RAG_RELATED_TYPES) or str(row.get("query_id", "")).startswith("T")


def reference_claims_for_query(
        query_id: str,
        row: Dict[str, Any],
        claims_by_qid: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    claims = [claim.get("required_claim", "") for claim in claims_by_qid.get(query_id, []) if
              claim.get("required_claim")]
    if claims:
        return claims

    reference = row.get("clean_reference_answer", "")
    sentences = split_sentences(reference)
    return sentences or ([reference] if reference else [])


def rag_context_precision(contexts: List[str], question: str, reference_claims: List[str]) -> Optional[float]:
    if not contexts:
        return None

    target = question + "\n" + "\n".join(reference_claims)
    relevance_flags = [
        lexical_overlap_score(target, context) >= CONTEXT_RELEVANCE_THRESHOLD
        for context in contexts
    ]

    relevant_count = sum(relevance_flags)
    if relevant_count == 0:
        return 0.0

    precision_sum = 0.0
    running_relevant = 0
    for index, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            running_relevant += 1
            precision_sum += running_relevant / index

    return round(precision_sum / relevant_count, 3)


def rag_context_recall(contexts: List[str], reference_claims: List[str]) -> Optional[float]:
    if not contexts or not reference_claims:
        return None

    context_text = "\n".join(contexts)
    supported = sum(
        1 for claim in reference_claims
        if lexical_overlap_score(claim, context_text) >= CONTEXT_RELEVANCE_THRESHOLD
    )

    return round(supported / len(reference_claims), 3)


def rag_answer_relevance(question: str, answer: str, reference_claims: List[str]) -> Optional[float]:
    if not answer:
        return None

    question_score = lexical_overlap_score(question, answer) if question else 0.0
    claims_text = "\n".join(reference_claims)
    claim_score = lexical_overlap_score(claims_text, answer) if claims_text else 0.0

    if question and claims_text:
        return round(0.4 * question_score + 0.6 * claim_score, 3)
    if question:
        return round(question_score, 3)
    if claims_text:
        return round(claim_score, 3)
    return None


def rag_faithfulness(answer: str, contexts: List[str]) -> Optional[Dict[str, Any]]:
    # True context-grounded faithfulness requires retrieved contexts.
    if not contexts:
        return None

    answer_claims = split_sentences(answer)
    if not answer_claims:
        return None

    context_text = "\n".join(contexts)
    claim_results = []
    supported_count = 0

    for claim in answer_claims:
        overlap = lexical_overlap_score(claim, context_text)
        supported = overlap >= FAITHFULNESS_SENTENCE_SUPPORT_THRESHOLD
        supported_count += int(supported)
        claim_results.append(
            {
                "answer_claim": claim,
                "support_overlap": round(overlap, 3),
                "supported": supported,
            }
        )

    faithfulness = supported_count / len(answer_claims)

    return {
        "faithfulness": round(faithfulness, 3),
        "supported_answer_claims": supported_count,
        "total_answer_claims": len(answer_claims),
        "support_source": "retrieved_contexts",
        "claim_support": claim_results,
    }


def rag_answer_correctness(
        claim_score: Optional[float],
        deterministic_score: Optional[float],
        answer_relevance: Optional[float],
        faithfulness_score: Optional[float],
) -> Optional[float]:
    components = []
    weights = []

    if claim_score is not None:
        components.append(claim_score)
        weights.append(0.55)

    if deterministic_score is not None:
        components.append(deterministic_score)
        weights.append(0.25)

    if answer_relevance is not None:
        components.append(answer_relevance)
        weights.append(0.20)

    if faithfulness_score is not None:
        components.append(faithfulness_score)
        weights.append(0.25)

    if not components:
        return None

    total_weight = sum(weights)
    return round(sum(value * weight for value, weight in zip(components, weights)) / total_weight, 3)


def rag_metrics_eval(
        query_id: str,
        row: Dict[str, Any],
        mas_output: Dict[str, Any],
        claims_by_qid: Dict[str, List[Dict[str, Any]]],
        claim_score: Optional[float],
        deterministic_score: Optional[float],
) -> Dict[str, Any]:
    question = mas_output.get("question", "")
    answer = mas_output.get("answer", "")
    contexts = mas_output.get("retrieved_contexts") or []
    ref_claims = reference_claims_for_query(query_id, row, claims_by_qid)

    context_precision = rag_context_precision(contexts, question, ref_claims)
    context_recall = rag_context_recall(contexts, ref_claims)
    answer_relevance = rag_answer_relevance(question, answer, ref_claims)

    faithfulness_details = rag_faithfulness(answer, contexts)
    faithfulness_score = faithfulness_details.get("faithfulness") if faithfulness_details else None
    unsupported_claim_rate = round(1 - faithfulness_score, 3) if faithfulness_score is not None else None

    context_sufficiency = (
        round(0.6 * context_recall + 0.4 * context_precision, 3)
        if context_recall is not None and context_precision is not None
        else None
    )

    answer_correctness = rag_answer_correctness(
        claim_score=claim_score,
        deterministic_score=deterministic_score,
        answer_relevance=answer_relevance,
        faithfulness_score=faithfulness_score,
    )

    composite_components = []
    composite_weights = []

    if claim_score is not None:
        composite_components.append(claim_score)
        composite_weights.append(0.55)

    if answer_relevance is not None:
        composite_components.append(answer_relevance)
        composite_weights.append(0.20)

    if faithfulness_score is not None:
        composite_components.append(faithfulness_score)
        composite_weights.append(0.25)

    if context_recall is not None:
        composite_components.append(context_recall)
        composite_weights.append(0.15)

    if context_precision is not None:
        composite_components.append(context_precision)
        composite_weights.append(0.10)

    rag_composite = None
    if composite_components:
        total_weight = sum(composite_weights)
        rag_composite = round(
            sum(value * weight for value, weight in zip(composite_components, composite_weights)) / total_weight,
            3,
        )

    return {
        "is_rag_related": is_rag_related(row),
        "contexts_available": bool(contexts),
        "num_contexts": len(contexts),
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_relevance": answer_relevance,
        "faithfulness": faithfulness_score,
        "faithfulness_details": faithfulness_details,
        "answer_correctness": answer_correctness,
        "completeness": claim_score,
        "unsupported_claim_rate": unsupported_claim_rate,
        "context_sufficiency": context_sufficiency,
        "rag_composite": rag_composite,
        "rag_note": (
            "retrieved_contexts are missing; Context Precision, Context Recall, Context Sufficiency, "
            "and true context-grounded Faithfulness are unavailable for this row."
            if not contexts else
            "RAG-style metrics computed using retrieved_contexts with transparent lexical-overlap heuristics."
        ),
    }


# ============================================================
# EVALUATION TYPE AND HARD CHECKS
# ============================================================

def requires_deterministic_eval(evaluation_type: str) -> bool:
    text = str(evaluation_type).lower()
    deterministic_markers = [
        "numeric", "exact", "ranking", "top-k", "correlation", "formula",
        "group proportion", "list", "entity", "set", "mean", "standard deviation",
    ]
    return any(marker in text for marker in deterministic_markers)


def is_open_ended_eval(evaluation_type: str, agent_category: str) -> bool:
    text = f"{evaluation_type} {agent_category}".lower()
    return any(marker in text for marker in OPEN_ENDED_TYPES)


def semantic_similarity(reference: str, answer: str) -> float:
    return SequenceMatcher(None, normalize_text(reference), normalize_text(answer)).ratio()


def verdict_from_score(score: float) -> str:
    if score >= PASS_THRESHOLD:
        return "PASS"
    if score >= PARTIAL_THRESHOLD:
        return "PARTIAL"
    return "FAIL"


def apply_hard_checks(
        query_id: str,
        answer: str,
        current_score: float,
        current_verdict: str,
) -> Tuple[float, str, List[str]]:
    answer_norm = normalize_text(answer)
    score = current_score
    verdict = current_verdict
    notes: List[str] = []

    if query_id == "D2" and "million" in answer_norm:
        score = min(score, 0.85)
        verdict = "PARTIAL"
        notes.append("Market Cap unit mismatch: answer uses million instead of B/billion.")

    if query_id == "R3" and "79.5" in answer_norm:
        score = min(score, 0.20)
        verdict = "FAIL"
        notes.append("Critical error: answer validates Samsung = 79.5, but corrected KTC Total Benchmark is 61.00.")

    if query_id == "P1":
        if "1.211" in answer_norm or "12.11" in answer_norm:
            score = min(score, 0.35)
            verdict = "FAIL"
            notes.append("Uses undocumented Remedy coefficient 1.211 / 12.11-point increase.")
        if "cannot provide a precise rank" in answer_norm or "would need" in answer_norm:
            score = min(score, 0.50)
            verdict = "FAIL" if score < PARTIAL_THRESHOLD else "PARTIAL"
            notes.append("Does not provide a valid requested 2027 rank and lacks rank-distribution assumptions.")

    if query_id == "P2":
        if "24.58" in answer_norm:
            score = 0.00
            verdict = "FAIL"
            notes.append("Uses four compounding periods; corrected 2025→2027 projection is 22.29.")
        elif "22.29" in answer_norm:
            score = max(score, 1.00)
            verdict = "PASS"

    if query_id == "P3":
        if "45" in answer_norm or "17.2" in answer_norm:
            score = 0.00
            verdict = "FAIL"
            notes.append("Incorrectly uses North America Purchasing Practices = 45 and inflated +17.2 result.")
        elif "0.46" in answer_norm and ("0.036" in answer_norm or "0.04" in answer_norm):
            score = max(score, 1.00)
            verdict = "PASS"

    if query_id == "P4":
        if "0.703" in answer_norm or "0.665" in answer_norm:
            score = min(score, 0.25)
            verdict = "FAIL"
            notes.append("Uses unsupported/inconsistent coefficient and does not define intervention delta.")
        if "cannot be quantified" in answer_norm or "without" in answer_norm and "defined" in answer_norm:
            score = max(score, 0.85)
            verdict = "PASS"

    if query_id in {"E3", "T4"}:
        if (
                ("asia" in answer_norm and "25" in answer_norm)
                and ("europe" in answer_norm and ("none" in answer_norm or "0" in answer_norm))
                and ("north america" in answer_norm and ("none" in answer_norm or "0" in answer_norm))
        ):
            score = max(score, 0.95)
            verdict = "PASS"
            notes.append("Diagnostic match for current UK MSA regional pattern; row remains NEEDS_VALIDATION.")
        elif "not available" in answer_norm or "no data available" in answer_norm or "cannot check" in answer_norm:
            score = 0.00
            verdict = "FAIL"
            notes.append("Failed to locate/parse UK MSA field; answer key expects regional UK MSA pattern.")

    if query_id == "S1" and "absence of data for traceability and purchasing practices" in answer_norm:
        score = min(score, 0.60)
        verdict = "PARTIAL"
        notes.append("Incorrectly claims Traceability/Purchasing Practices data are absent.")

    if query_id == "R1":
        if "specific countries were not explicitly mentioned" in answer_norm:
            score = min(score, 0.60)
            verdict = "PARTIAL"
            notes.append("Answer mentions China/Malaysia but fails PDF-specific extraction requirement.")
        elif "ongoing concerns" in answer_norm or "recent ilo resources" in answer_norm:
            score = min(score, 0.60)
            verdict = "PARTIAL"
            notes.append("External validation is too vague; exact ILO source/date/statistic is not snapshotted.")

    if query_id == "R4" and ("units not specified" in answer_norm or "recent web sources indicates" in answer_norm):
        score = min(score, 0.55)
        verdict = "FAIL" if score < PARTIAL_THRESHOLD else "PARTIAL"
        notes.append("External comparison lacks reproducible benchmark, date, units, and source details.")

    if query_id == "R2" and "ilo indicators of forced labor 2025" in answer_norm:
        score = min(score, 0.60)
        verdict = "PARTIAL"
        notes.append("External ILO report/source metadata is not sufficiently verified or snapshotted.")

    return round(score, 3), verdict, notes


# ============================================================
# OPTIONAL LLM-AS-A-JUDGE
# ============================================================

def maybe_run_llm_judge(
        results: List[Dict[str, Any]],
        answer_key_rows: Dict[str, Dict[str, Any]],
        model: str,
) -> List[Dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install OpenAI client first: pip install openai") from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()

    for result in results:
        row = answer_key_rows.get(result["query_id"], {})
        if not is_open_ended_eval(row.get("evaluation_type", ""), row.get("agent_category", "")):
            continue

        prompt = {
            "query_id": result["query_id"],
            "question": result.get("question", ""),
            "evaluation_type": row.get("evaluation_type", ""),
            "reference_answer": row.get("clean_reference_answer", ""),
            "evaluation_criteria": row.get("pass_criteria", ""),
            "retrieved_contexts": result.get("retrieved_contexts", []),
            "mas_answer": result.get("answer", ""),
            "instruction": (
                "Evaluate strictly using only the reference answer, required claims, and retrieved contexts. "
                "Return valid JSON with: claim_level_correctness, completeness, faithfulness, answer_relevance, "
                "unsupported_claim_rate, ethical_adequacy, final_verdict, reason."
            ),
        }

        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": "You are a strict academic evaluator for MAS answer quality. Use only provided evidence.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0,
        )

        raw = response.output_text.strip()

        try:
            result["llm_judge"] = json.loads(raw)
        except json.JSONDecodeError:
            result["llm_judge"] = {"raw_judge_output": raw}

    return results


# ============================================================
# MAIN EVALUATION
# ============================================================

def combine_scores(
        deterministic_score: Optional[float],
        claim_score: Optional[float],
        rag_composite: Optional[float],
        fallback: Optional[float],
        row: Dict[str, Any],
) -> float:
    scores = []

    if deterministic_score is not None:
        scores.append(deterministic_score)

    if claim_score is not None:
        scores.append(claim_score)

    # Use RAG composite only as an additional signal. It is not allowed to rescue a poor deterministic/claim score.
    if is_rag_related(row) and rag_composite is not None:
        scores.append(rag_composite)

    if scores:
        return round(min(scores), 3)

    return round(fallback if fallback is not None else 0.0, 3)


def evaluate_single_query(
        row: Dict[str, Any],
        mas_output: Optional[Dict[str, Any]],
        claims_by_qid: Dict[str, List[Dict[str, Any]]],
        global_entities: List[str],
) -> Dict[str, Any]:
    query_id = row["query_id"]
    answer_key_status = row.get("status")
    included_in_final_accuracy = answer_key_status in FINAL_KEY_STATUSES

    if mas_output is None:
        return {
            "query_id": query_id,
            "query_number": None,
            "agent_category": row.get("agent_category"),
            "answer_key_status": answer_key_status,
            "included_in_final_accuracy": included_in_final_accuracy,
            "adjudication_scope": "FINAL" if included_in_final_accuracy else "DIAGNOSTIC_ONLY",
            "evaluation_type": row.get("evaluation_type"),
            "mas_status": "MISSING",
            "latency_seconds": None,
            "score": 0.0,
            "verdict": "MISSING_OUTPUT",
            "notes": "No MAS output found for this query.",
            "question": row.get("query_text", ""),
            "answer": "",
            "retrieved_contexts": [],
            "details": {},
        }

    answer = mas_output.get("answer", "")
    evaluation_type = row.get("evaluation_type", "")
    details: Dict[str, Any] = {}
    notes: List[str] = []

    deterministic_score = None
    if requires_deterministic_eval(evaluation_type):
        det_details, deterministic_score = deterministic_structured_eval(row, answer, global_entities)
        details["deterministic_eval"] = det_details

    claim_score = None
    claim_eval = score_atomic_claims(query_id, answer, claims_by_qid)
    if claim_eval:
        details["claim_eval"] = claim_eval
        claim_score = claim_eval.get("claim_score")

    rag_composite = None
    if is_rag_related(row):
        rag_details = rag_metrics_eval(
            query_id=query_id,
            row=row,
            mas_output=mas_output,
            claims_by_qid=claims_by_qid,
            claim_score=claim_score,
            deterministic_score=deterministic_score,
        )
        details["rag_eval"] = rag_details
        rag_composite = rag_details.get("rag_composite")

    fallback = None
    if deterministic_score is None and claim_score is None and rag_composite is None:
        fallback = semantic_similarity(row.get("clean_reference_answer", ""), answer)
        details["fallback_lexical_similarity"] = round(fallback, 3)

    final_score = combine_scores(
        deterministic_score=deterministic_score,
        claim_score=claim_score,
        rag_composite=rag_composite,
        fallback=fallback,
        row=row,
    )

    verdict = verdict_from_score(final_score)
    final_score, verdict, hard_notes = apply_hard_checks(query_id, answer, final_score, verdict)
    notes.extend(hard_notes)

    if not included_in_final_accuracy:
        notes.append(
            f"Answer-key status is {answer_key_status}; this row is diagnostic only and excluded from final accuracy."
        )

    return {
        "query_id": query_id,
        "query_number": mas_output.get("query_number"),
        "agent_category": row.get("agent_category"),
        "answer_key_status": answer_key_status,
        "included_in_final_accuracy": included_in_final_accuracy,
        "adjudication_scope": "FINAL" if included_in_final_accuracy else "DIAGNOSTIC_ONLY",
        "evaluation_type": evaluation_type,
        "mas_status": mas_output.get("mas_status"),
        "latency_seconds": mas_output.get("latency_seconds"),
        "score": round(float(final_score), 3),
        "verdict": verdict,
        "notes": " ".join(notes),
        "question": mas_output.get("question", ""),
        "answer": answer,
        "retrieved_contexts": mas_output.get("retrieved_contexts", []),
        "details": details,
    }


def evaluate_all(answer_key: Dict[str, Any], mas_outputs: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = answer_key.get("answer_key", [])
    rows_by_qid = {row["query_id"]: row for row in rows}
    claims_by_qid = build_claim_index(answer_key)
    global_entities = build_global_entity_universe(answer_key)

    results = []
    for query_id in QUERY_ID_ORDER:
        if query_id not in rows_by_qid:
            continue

        result = evaluate_single_query(
            row=rows_by_qid[query_id],
            mas_output=mas_outputs.get(query_id),
            claims_by_qid=claims_by_qid,
            global_entities=global_entities,
        )
        results.append(result)

    return results


# ============================================================
# SUMMARIES AND OUTPUT WRITERS
# ============================================================

def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((p / 100) * (len(values) - 1))))
    return values[index]


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "pass": 0,
            "partial": 0,
            "fail": 0,
            "missing": 0,
            "strict_pass_rate": 0,
            "pass_or_partial_rate": 0,
            "mean_score": 0,
            "mean_latency_seconds": None,
            "median_latency_seconds": None,
            "p95_latency_seconds": None,
        }

    latencies = [
        row["latency_seconds"]
        for row in rows
        if isinstance(row.get("latency_seconds"), (int, float))
    ]

    n = len(rows)
    pass_count = sum(row["verdict"] == "PASS" for row in rows)
    partial_count = sum(row["verdict"] == "PARTIAL" for row in rows)
    fail_count = sum(row["verdict"] == "FAIL" for row in rows)
    missing_count = sum(row["verdict"] == "MISSING_OUTPUT" for row in rows)
    mean_score = sum(float(row["score"]) for row in rows) / n

    return {
        "n": n,
        "pass": pass_count,
        "partial": partial_count,
        "fail": fail_count,
        "missing": missing_count,
        "strict_pass_rate": round(pass_count / n, 3),
        "pass_or_partial_rate": round((pass_count + partial_count) / n, 3),
        "mean_score": round(mean_score, 3),
        "mean_latency_seconds": round(statistics.mean(latencies), 3) if latencies else None,
        "median_latency_seconds": round(statistics.median(latencies), 3) if latencies else None,
        "p95_latency_seconds": round(percentile(latencies, 95), 3) if latencies else None,
    }


def summarize_by_category(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    categories = sorted({row.get("agent_category", "UNKNOWN") for row in rows})
    return {
        category: summarize_results([row for row in rows if row.get("agent_category") == category])
        for category in categories
    }


def summarize_diagnostic_statuses(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("answer_key_status", "UNKNOWN"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def write_main_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "query_id",
        "query_number",
        "agent_category",
        "answer_key_status",
        "included_in_final_accuracy",
        "adjudication_scope",
        "evaluation_type",
        "mas_status",
        "latency_seconds",
        "score",
        "verdict",
        "notes",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_category_csv(summary: Dict[str, Dict[str, Any]], path: Path) -> None:
    fields = [
        "agent_category",
        "n",
        "pass",
        "partial",
        "fail",
        "missing",
        "strict_pass_rate",
        "pass_or_partial_rate",
        "mean_score",
        "mean_latency_seconds",
        "median_latency_seconds",
        "p95_latency_seconds",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for category, metrics in summary.items():
            row = {"agent_category": category}
            row.update(metrics)
            writer.writerow({field: row.get(field, "") for field in fields})


def write_rag_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "query_id",
        "agent_category",
        "answer_key_status",
        "contexts_available",
        "num_contexts",
        "context_precision",
        "context_recall",
        "answer_relevance",
        "faithfulness",
        "answer_correctness",
        "completeness",
        "unsupported_claim_rate",
        "context_sufficiency",
        "rag_composite",
        "rag_note",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            rag = row.get("details", {}).get("rag_eval")
            if not rag:
                continue
            out = {
                "query_id": row.get("query_id"),
                "agent_category": row.get("agent_category"),
                "answer_key_status": row.get("answer_key_status"),
            }
            out.update({field: rag.get(field) for field in fields if field not in out})
            writer.writerow({field: out.get(field, "") for field in fields})


# ============================================================
# ENTRYPOINT
# ============================================================

def main() -> None:
    ensure_output_dir()

    answer_key_path = first_existing_path(ANSWER_KEY_CANDIDATES, "Answer key JSON")
    generated_answers_path = first_existing_path(GENERATED_ANSWERS_CANDIDATES, "Generated answers JSON/log")

    print(f"Using answer key: {answer_key_path}")
    print(f"Using generated answers: {generated_answers_path}")

    answer_key = json.loads(read_text(answer_key_path))
    mas_outputs = load_mas_outputs(generated_answers_path)

    print(f"Parsed MAS outputs: {len(mas_outputs)}")

    results = evaluate_all(answer_key, mas_outputs)
    answer_key_rows = {row["query_id"]: row for row in answer_key.get("answer_key", [])}

    if USE_LLM_JUDGE:
        results = maybe_run_llm_judge(results, answer_key_rows, JUDGE_MODEL)

    final_rows = [row for row in results if row["included_in_final_accuracy"]]
    diagnostic_only_rows = [row for row in results if not row["included_in_final_accuracy"]]

    summary = {
        "final_accuracy_only_validated_answer_key_rows": summarize_results(final_rows),
        "diagnostic_outcomes_all_28_rows_not_final_accuracy": summarize_results(results),
        "diagnostic_only_rows_excluded_from_final_accuracy": summarize_results(diagnostic_only_rows),
        "by_category_final_rows": summarize_by_category(final_rows),
        "by_category_all_rows_diagnostic": summarize_by_category(results),
        "answer_key_status_counts": summarize_diagnostic_statuses(results),
        "important_note": (
            "Only rows with CLEAN, CLEAN_WITH_RUBRIC, or CLEAN_WITH_METHOD_NOTE are included in final accuracy. "
            "Rows marked NEEDS_CORRECTION, NEEDS_SOURCE_SNAPSHOT, or NEEDS_VALIDATION are diagnostic only. "
            "Context Precision, Context Recall, Context Sufficiency, and true context-grounded Faithfulness are "
            "computed only when retrieved_contexts are present."
        ),
    }

    report = {
        "metadata": {
            "answer_key_path": str(answer_key_path),
            "generated_answers_path": str(generated_answers_path),
            "query_id_order": QUERY_ID_ORDER,
            "final_key_statuses": sorted(FINAL_KEY_STATUSES),
            "diagnostic_only_statuses": sorted(DIAGNOSTIC_ONLY_STATUSES),
            "pass_threshold": PASS_THRESHOLD,
            "partial_threshold": PARTIAL_THRESHOLD,
            "llm_judge_used": USE_LLM_JUDGE,
            "judge_model": JUDGE_MODEL if USE_LLM_JUDGE else None,
            "metrics_included": [
                "Operational Completion",
                "Latency when available",
                "Numeric Tolerance Match",
                "Numeric Precision/Recall/F1",
                "Entity Precision/Recall/F1",
                "Entity-Value Pair Accuracy",
                "Unit Consistency Check",
                "Top-K / Ranking Accuracy where applicable",
                "Claim-Level Coverage",
                "Completeness",
                "Answer Relevance heuristic",
                "Context Precision heuristic only when retrieved_contexts exist",
                "Context Recall heuristic only when retrieved_contexts exist",
                "Faithfulness/Groundedness heuristic only when retrieved_contexts exist",
                "Answer Correctness hybrid heuristic",
                "Critical Error Rule Checks",
                "PASS/PARTIAL/FAIL Classification",
                "Strict Pass Rate",
                "Pass-or-Partial Rate",
                "Category-wise Summary",
                "Optional LLM-as-a-Judge",
            ],
        },
        "summary": summary,
        "results": results,
    }

    OUTPUT_JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_main_csv(results, OUTPUT_CSV_PATH)
    write_category_csv(summary["by_category_all_rows_diagnostic"], OUTPUT_CATEGORY_CSV_PATH)
    write_rag_csv(results, OUTPUT_RAG_CSV_PATH)

    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote JSON report: {OUTPUT_JSON_PATH}")
    print(f"Wrote CSV summary: {OUTPUT_CSV_PATH}")
    print(f"Wrote category summary: {OUTPUT_CATEGORY_CSV_PATH}")
    print(f"Wrote RAG metrics: {OUTPUT_RAG_CSV_PATH}")


if __name__ == "__main__":
    main()
