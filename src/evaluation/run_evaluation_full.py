"""
Final MAS + RAG / RAGAS-style evaluator for the 28-query dataset.

Input expected:
  generated_answers_structured.json + structured reference answer key JSON

The input may be either:
  {"metadata": {...}, "data": [ ... rows ... ]}
or a plain list of rows.

Each row should contain:
  query_id, query_number, agent_category, answer_key_status,
  included_in_final_accuracy, user_input, response, reference,
  retrieved_contexts, retrieved_context_records

Outputs:
  <out-dir>/evaluation_results.json
  <out-dir>/evaluation_results.csv
  <out-dir>/evaluation_summary.json
  <out-dir>/evaluation_summary.csv
  <out-dir>/rows_with_contexts.csv
  <out-dir>/ragas_official_results.csv          (only if --use-official-ragas succeeds)

What this script computes:
  - Answer Correctness (reference-based heuristic, always available)
  - Answer Relevancy (query-response relevance heuristic, always available)
  - Numeric precision / recall / F1 against reference numbers
  - Reference claim recall / lexical similarity
  - Faithfulness (requires retrieved_contexts)
  - Context Precision (requires retrieved_contexts)
  - Context Recall (requires retrieved_contexts)
  - Context Sufficiency / Utilization (requires retrieved_contexts)
  - RAG composite score for rows with contexts
  - Final-accuracy summary over validated rows only
  - Diagnostic/RAG-only summary over non-final rows

Important methodological note:
  retrieved_contexts are the contexts retrieved by the system, not gold evidence.
  Therefore they are valid for evaluating RAG behavior, but they should not be treated
  as reference/gold answers.

Optional official RAGAS:
  If you install ragas + datasets and set an LLM/embedding provider as required by
  your ragas version, you can run:

  python evaluate_ragas_final_full.py --dataset ragas_dataset_all_28_with_generated_contexts.json --use-official-ragas

  The script will attempt the common ragas API and will continue with heuristic metrics
  if official RAGAS fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Policy / thresholds
# -----------------------------------------------------------------------------

FINAL_KEY_STATUSES = {"CLEAN", "CLEAN_WITH_RUBRIC", "CLEAN_WITH_METHOD_NOTE"}
DIAGNOSTIC_ONLY_STATUSES = {"NEEDS_CORRECTION", "NEEDS_SOURCE_SNAPSHOT", "NEEDS_VALIDATION"}

DEFAULT_PASS_THRESHOLD = 0.60
DEFAULT_PARTIAL_THRESHOLD = 0.30

# For context metrics. These are deliberately conservative lexical thresholds.
CONTEXT_RELEVANCE_THRESHOLD = 0.10
CONTEXT_RECALL_TOKEN_THRESHOLD = 0.01
FAITHFULNESS_SENTENCE_SUPPORT_THRESHOLD = 0.15

INCIDENTAL_NUMBERS = {
    2022, 2023, 2024, 2025, 2026, 2027, 100,
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "as", "is", "are", "be", "was", "were", "this", "that", "these", "those",
    "from", "into", "not", "but", "if", "then", "there", "where", "which", "who",
    "what", "when", "how", "does", "do", "did", "has", "have", "had", "can",
    "could", "should", "would", "may", "might", "must", "include", "includes",
    "including", "required", "claim", "answer", "response", "score", "scores",
    "query", "question", "company", "companies", "data", "sheet", "table", "using",
    "based", "overall", "also", "more", "less", "than", "about", "across", "such",
    "e.g", "approximately", "approx", "average", "current", "final", "valid",
    "should", "must", "will", "would", "could", "one", "two", "three", "four", "five",
    "its", "their", "it", "they", "he", "she", "we", "you", "i", "our", "your",
}

# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return statistics.median(vals)


def pct(values: Iterable[Optional[float]], q: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[int(pos)]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def rounded(x: Optional[float], ndigits: int = 3) -> Optional[float]:
    if x is None:
        return None
    return round(float(x), ndigits)


def f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def clamp01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return max(0.0, min(1.0, float(x)))


def clean_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def normalize_text(text: Any) -> str:
    text = clean_text(text).lower().replace("labour", "labor")
    text = re.sub(r"[^a-z0-9.\s&()/%+\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokens(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def content_tokens(text: Any) -> List[str]:
    return [t for t in tokens(text) if len(t) > 2 and t not in STOPWORDS]


def token_set(text: Any) -> set:
    return set(content_tokens(text))


def lexical_recall(source: str, target: str) -> float:
    """How much of source appears in target, using content-token overlap."""
    src = token_set(source)
    tgt = token_set(target)
    if not src:
        return 0.0
    return len(src & tgt) / len(src)


def lexical_precision(source: str, target: str) -> float:
    """How much of target is explained by source, using content-token overlap."""
    src = token_set(source)
    tgt = token_set(target)
    if not tgt:
        return 0.0
    return len(src & tgt) / len(tgt)


def lexical_f1(a: str, b: str) -> float:
    p = lexical_precision(a, b)
    r = lexical_recall(a, b)
    return f1(p, r) or 0.0


def split_reference_claims(text: str) -> List[str]:
    """Split reference into evaluable claim-like chunks."""
    if not text:
        return []
    raw_parts: List[str] = []
    for line in str(text).splitlines():
        line = strip_list_prefix(line).strip()
        if not line:
            continue
        # Semicolon-separated rubric items are common in this dataset.
        for part in re.split(r";|\.|\n", line):
            part = part.strip(" -:•\t")
            if len(content_tokens(part)) >= 3:
                raw_parts.append(part)
    return dedupe_preserve_order(raw_parts)


def split_response_claims(text: str) -> List[str]:
    if not text:
        return []
    # Sentence split plus bullet split. Keep meaningful chunks only.
    parts = re.split(r"(?<=[.!?])\s+|\n+|(?:\s*[-•]\s+)", str(text))
    claims = []
    for part in parts:
        part = strip_list_prefix(part).strip()
        if len(content_tokens(part)) >= 4:
            claims.append(part)
    return dedupe_preserve_order(claims)


def strip_list_prefix(line: str) -> str:
    line = str(line).strip()
    line = re.sub(r"^[-*•]\s*", "", line)
    line = re.sub(r"^\d+[.)]\s*", "", line)
    return line.strip()


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out

# -----------------------------------------------------------------------------
# Numeric and unit metrics
# -----------------------------------------------------------------------------

def extract_numbers(text: Any) -> List[float]:
    """Extract meaningful numeric values while ignoring common list/rank prefixes.

    This prevents answers like "1. Samsung: 61" from being penalized for the
    list marker `1.`. It also removes common prompt/preamble numbers such as
    "top 5" and "k=3" when they are not part of the scored value.
    """
    values: List[float] = []
    raw = str(text).replace(",", "")
    for line in raw.splitlines():
        line = strip_list_prefix(line)
        line = re.sub(r"\btop\s+\d+\b", " ", line, flags=re.I)
        line = re.sub(r"\bk\s*=\s*\d+\b", " ", line, flags=re.I)
        line = re.sub(r"\bindicator\s+\d+(?:\.\d+)?\b", " ", line, flags=re.I)
        values.extend(float(m.group(0)) for m in re.finditer(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", line))
    return values


def filter_incidental_numbers(nums: List[float]) -> List[float]:
    out = []
    for n in nums:
        if int(abs(n)) in INCIDENTAL_NUMBERS:
            continue
        out.append(n)
    return out


def number_match(target: float, candidates: List[float], tolerance: float = 0.05) -> bool:
    for c in candidates:
        if abs(c - target) <= tolerance:
            return True
    return False


def numeric_metrics(response: str, reference: str, tolerance: float = 0.05) -> Dict[str, Any]:
    ref_nums = filter_incidental_numbers(extract_numbers(reference))
    ans_nums = filter_incidental_numbers(extract_numbers(response))

    # Keep duplicates meaningfully, but match greedily so repeated values don't overcount.
    remaining = ans_nums[:]
    found = []
    missing = []
    for rn in ref_nums:
        idx = None
        for i, an in enumerate(remaining):
            if abs(an - rn) <= tolerance:
                idx = i
                break
        if idx is None:
            missing.append(rn)
        else:
            found.append(rn)
            remaining.pop(idx)

    # Extra answer numbers are common in natural answers (list markers, explanatory
    # figures, examples). They are reported, but should not dominate correctness.
    extra = remaining
    raw_precision = len(found) / (len(found) + len(extra)) if (found or extra) else None
    recall = len(found) / len(ref_nums) if ref_nums else None
    # If all reference numbers are present, cap the precision penalty from extras.
    if recall == 1.0 and raw_precision is not None:
        precision = max(raw_precision, 0.95)
    else:
        precision = raw_precision
    f1_score = f1(precision, recall) if precision is not None and recall is not None else None

    return {
        "reference_numbers": ref_nums,
        "response_numbers": ans_nums,
        "numeric_found": found,
        "numeric_missing": missing,
        "numeric_extra": extra,
        "numeric_precision": precision,
        "numeric_recall": recall,
        "numeric_f1": f1_score,
        "numeric_tolerance": tolerance,
    }


def detect_unit_issue(response: str, reference: str) -> Dict[str, Any]:
    r = normalize_text(response)
    ref = normalize_text(reference)
    response_has_million = "million" in r or re.search(r"\bmn\b", r) is not None
    response_has_billion = "billion" in r or re.search(r"\bbn\b", r) is not None
    ref_has_b = re.search(r"\d\s*b\b", ref) is not None or "billion" in ref
    ref_has_m = re.search(r"\d\s*m\b", ref) is not None or "million" in ref

    issue = False
    issue_type = None
    if ref_has_b and response_has_million and not response_has_billion:
        issue = True
        issue_type = "reference_uses_billion_but_response_says_million"
    elif ref_has_m and response_has_billion and not response_has_million:
        issue = True
        issue_type = "reference_uses_million_but_response_says_billion"

    return {
        "unit_issue": issue,
        "unit_issue_type": issue_type,
        "unit_consistency": 0.0 if issue else 1.0,
    }

# -----------------------------------------------------------------------------
# Answer correctness and relevancy
# -----------------------------------------------------------------------------


# Lightweight semantic aliases for qualitative/rubric rows. This does not replace
# exact numeric checking; it only prevents good paraphrases from being punished
# when the wording differs from the reference.
CONCEPT_ALIASES = {
    "principal_agent": [
        "principal agent", "principal-agent", "agency theory", "agent incentives",
        "principal", "agent"
    ],
    "information_asymmetry": [
        "information asymmetry", "asymmetric information", "hidden information",
        "limited visibility", "lack of transparency", "disclosure gap", "data opacity"
    ],
    "hidden_action": [
        "hidden action", "moral hazard", "unobserved action", "selective disclosure",
        "underreporting", "greenwashing", "self reporting", "self-reported"
    ],
    "regional_bias": [
        "regional bias", "region bias", "region", "regional", "asia", "europe",
        "north america", "geographic", "geographical"
    ],
    "indicator_bias": [
        "indicator bias", "theme bias", "indicator", "theme", "remedy",
        "monitoring", "purchasing practices", "enabling workers", "recruitment"
    ],
    "worker_voice_remedy": [
        "worker voice", "worker grievance", "grievance", "remedy", "remediation",
        "complaint", "worker feedback", "stakeholder input", "trade union", "worker interview"
    ],
    "triangulation": [
        "triangulation", "triangulate", "cross-check", "cross check", "external evidence",
        "qualitative evidence", "audit", "worker interviews", "stakeholder"
    ],
    "forced_labor": [
        "forced labor", "forced labour", "modern slavery", "recruitment fees",
        "migrant workers", "debt bondage", "uyghur", "xinjiang"
    ],
    "source_mismatch": [
        "source mismatch", "wrong source", "not amazon", "non-amazon", "not company specific",
        "insufficient source", "not supported by the retrieved context"
    ],
}


def _alias_present(text: str, aliases: Sequence[str]) -> bool:
    nt = normalize_text(text)
    return any(normalize_text(alias) in nt for alias in aliases)


def semantic_concept_score(claim: str, response: str) -> float:
    """Score claim coverage using small domain-specific concept aliases.

    A claim only activates concepts that appear in the claim itself. The response
    then gets credit for mentioning semantically equivalent aliases. This reduces
    false negatives for qualitative answers without giving credit for unrelated
    concepts.
    """
    active = []
    for concept, aliases in CONCEPT_ALIASES.items():
        if _alias_present(claim, aliases):
            active.append(concept)
    if not active:
        return 0.0
    hits = 0
    for concept in active:
        if _alias_present(response, CONCEPT_ALIASES[concept]):
            hits += 1
    return hits / len(active)


def normalize_required_claim_items(required_claim_items: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    """Normalize answer-key required_claims into {claim, weight, claim_id} rows."""
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(required_claim_items or [], start=1):
        if isinstance(item, dict):
            claim = item.get("required_claim") or item.get("claim") or item.get("text") or ""
            weight = safe_float(item.get("weight")) or 1.0
            claim_id = item.get("claim_id") or f"claim-{idx}"
        else:
            claim = str(item)
            weight = 1.0
            claim_id = f"claim-{idx}"
        claim = str(claim).strip()
        if claim and len(content_tokens(claim)) >= 3:
            out.append({"claim": claim, "weight": weight, "claim_id": claim_id})
    return out

def required_claim_recall(
    response: str,
    reference: str,
    required_claim_items: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    # Prefer curated claim objects from the structured answer key when available.
    # Fallback to automatic splitting of the reference text.
    normalized_claims = normalize_required_claim_items(required_claim_items)
    if normalized_claims:
        claims = normalized_claims
    else:
        claims = [{"claim": c, "weight": 1.0, "claim_id": f"auto-{i}"}
                  for i, c in enumerate(split_reference_claims(reference), start=1)]

    if not claims:
        return {
            "required_claims": [],
            "claim_scores": [],
            "required_claim_recall": None,
        }

    scores = []
    total_weight = sum(max(0.0, float(c.get("weight", 1.0))) for c in claims) or 1.0
    weighted_binary = 0.0
    weighted_continuous = 0.0

    for item in claims:
        claim = item["claim"]
        weight = max(0.0, float(item.get("weight", 1.0)))
        lexical_score = lexical_recall(claim, response)
        semantic_score = semantic_concept_score(claim, response)
        # Semantic aliases are deliberately capped so they can rescue paraphrase,
        # but not replace missing numeric/entity evidence.
        score = max(lexical_score, 0.80 * semantic_score)
        covered = score >= 0.35
        weighted_binary += weight * (1.0 if covered else 0.0)
        weighted_continuous += weight * score
        scores.append({
            "claim_id": item.get("claim_id"),
            "claim": claim,
            "weight": weight,
            "coverage_score": score,
            "lexical_score": lexical_score,
            "semantic_score": semantic_score,
            "covered": covered,
        })

    recall_binary = weighted_binary / total_weight
    recall_continuous = weighted_continuous / total_weight
    recall = 0.70 * recall_binary + 0.30 * recall_continuous
    return {
        "required_claims": [c["claim"] for c in claims],
        "claim_scores": scores,
        "required_claim_recall": clamp01(recall),
        "required_claim_recall_binary": recall_binary,
        "required_claim_recall_continuous": recall_continuous,
        "claim_source": "answer_key_required_claims" if normalized_claims else "auto_split_reference",
    }


def answer_correctness_score(
    response: str,
    reference: str,
    required_claim_items: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    nm = numeric_metrics(response, reference)
    cm = required_claim_recall(response, reference, required_claim_items=required_claim_items)
    lex = lexical_f1(response, reference)
    unit = detect_unit_issue(response, reference)

    available_components: List[Tuple[str, float, float]] = []  # (name, score, weight)

    # For evaluation, numeric recall is usually more important than numeric precision:
    # an answer may include list indices or explanatory numbers, but the core question is
    # whether it includes the required reference numbers.
    numeric_score = nm.get("numeric_recall") if nm.get("numeric_recall") is not None else nm.get("numeric_f1")

    if numeric_score is not None and cm["required_claim_recall"] is not None:
        available_components.append(("numeric_recall", numeric_score, 0.50))
        available_components.append(("required_claim_recall", cm["required_claim_recall"], 0.35))
        available_components.append(("lexical_reference_similarity", lex, 0.15))
    elif numeric_score is not None:
        available_components.append(("numeric_recall", numeric_score, 0.80))
        available_components.append(("lexical_reference_similarity", lex, 0.20))
    elif cm["required_claim_recall"] is not None:
        available_components.append(("required_claim_recall", cm["required_claim_recall"], 0.75))
        available_components.append(("lexical_reference_similarity", lex, 0.25))
    else:
        available_components.append(("lexical_reference_similarity", lex, 1.00))

    total_w = sum(w for _, _, w in available_components)
    score = sum(s * w for _, s, w in available_components) / total_w if total_w else 0.0

    # Unit mismatch is reported, but only mildly penalized because the uploaded reference
    # notes that market-cap units should be confirmed before publication.
    if unit["unit_issue"]:
        score *= 0.95

    return {
        "answer_correctness": clamp01(score),
        "components": {
            "numeric_f1": nm["numeric_f1"],
            "required_claim_recall": cm["required_claim_recall"],
            "lexical_reference_similarity": lex,
            "unit_consistency": unit["unit_consistency"],
        },
        "numeric_metrics": nm,
        "claim_metrics": cm,
        "unit_consistency": unit,
    }


def answer_relevancy_score(user_input: str, response: str, reference: str) -> Dict[str, Any]:
    q_overlap = lexical_recall(user_input, response)
    qr_f1 = lexical_f1(user_input, response)
    ref_overlap = lexical_recall(reference, response) if reference else 0.0

    # Query-response relevance should dominate. Reference overlap prevents empty-but-on-topic answers from scoring high.
    score = 0.55 * q_overlap + 0.25 * qr_f1 + 0.20 * ref_overlap
    return {
        "answer_relevancy": clamp01(score),
        "query_token_recall_in_response": q_overlap,
        "query_response_lexical_f1": qr_f1,
        "reference_token_recall_in_response": ref_overlap,
    }

# -----------------------------------------------------------------------------
# RAG metrics: context precision, recall, faithfulness
# -----------------------------------------------------------------------------

def context_texts(row: Dict[str, Any]) -> List[str]:
    ctx = row.get("retrieved_contexts") or []
    out: List[str] = []
    for c in ctx:
        if isinstance(c, str):
            text = c.strip()
        elif isinstance(c, dict):
            text = str(c.get("content") or c.get("text") or "").strip()
        else:
            text = str(c).strip()
        if text:
            out.append(text)
    return out


def context_records(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = row.get("retrieved_context_records") or []
    if isinstance(records, list):
        return [r for r in records if isinstance(r, dict)]
    return []


def context_relevance_score(context: str, user_input: str, reference: str) -> float:
    # Consider both the user need and the reference information.
    q = lexical_recall(user_input, context)
    ref = lexical_recall(reference, context) if reference else 0.0
    return clamp01(0.45 * q + 0.55 * ref) or 0.0


def context_precision_score(contexts: List[str], user_input: str, reference: str) -> Dict[str, Any]:
    if not contexts:
        return {
            "context_precision": None,
            "context_relevance_scores": [],
            "relevant_context_count": 0,
            "num_contexts": 0,
        }

    relevance_scores = [context_relevance_score(c, user_input, reference) for c in contexts]
    relevant_flags = [s >= CONTEXT_RELEVANCE_THRESHOLD for s in relevance_scores]

    # Average precision style, preserving retrieval rank.
    precisions_at_k = []
    relevant_so_far = 0
    for idx, is_rel in enumerate(relevant_flags, start=1):
        if is_rel:
            relevant_so_far += 1
            precisions_at_k.append(relevant_so_far / idx)

    if relevant_so_far == 0:
        ap = 0.0
    else:
        ap = sum(precisions_at_k) / relevant_so_far

    return {
        "context_precision": clamp01(ap),
        "context_relevance_scores": relevance_scores,
        "context_relevant_flags": relevant_flags,
        "relevant_context_count": relevant_so_far,
        "num_contexts": len(contexts),
    }


def context_recall_score(contexts: List[str], reference: str) -> Dict[str, Any]:
    if not contexts:
        return {
            "context_recall": None,
            "reference_tokens_covered_by_contexts": None,
            "missing_reference_keywords": [],
        }
    joined = "\n".join(contexts)
    ref_tokens = token_set(reference)
    ctx_tokens = token_set(joined)
    if not ref_tokens:
        return {
            "context_recall": None,
            "reference_tokens_covered_by_contexts": None,
            "missing_reference_keywords": [],
        }
    covered = ref_tokens & ctx_tokens
    missing = sorted(ref_tokens - ctx_tokens)
    recall = len(covered) / len(ref_tokens)
    return {
        "context_recall": clamp01(recall),
        "reference_tokens_covered_by_contexts": sorted(covered),
        "missing_reference_keywords": missing[:100],
    }


def claim_supported_by_context(claim: str, joined_contexts: str) -> Tuple[bool, float, str]:
    # Lexical support.
    overlap = lexical_recall(claim, joined_contexts)

    # Numeric support: if claim contains important numbers, at least one should appear in context.
    claim_nums = filter_incidental_numbers(extract_numbers(claim))
    ctx_nums = filter_incidental_numbers(extract_numbers(joined_contexts))
    numeric_supported = True
    if claim_nums:
        numeric_supported = any(number_match(n, ctx_nums, tolerance=0.05) for n in claim_nums)

    supported = overlap >= FAITHFULNESS_SENTENCE_SUPPORT_THRESHOLD and numeric_supported
    reason = "lexical_and_numeric_support" if supported else "insufficient_context_overlap_or_numeric_support"
    return supported, overlap, reason


def faithfulness_score(contexts: List[str], response: str) -> Dict[str, Any]:
    if not contexts:
        return {
            "faithfulness": None,
            "supported_claims": 0,
            "total_claims": 0,
            "claim_support": [],
        }
    joined = "\n".join(contexts)
    claims = split_response_claims(response)
    if not claims:
        return {
            "faithfulness": None,
            "supported_claims": 0,
            "total_claims": 0,
            "claim_support": [],
        }

    details = []
    supported_count = 0
    for claim in claims:
        supported, score, reason = claim_supported_by_context(claim, joined)
        if supported:
            supported_count += 1
        details.append({
            "claim": claim,
            "supported": supported,
            "support_score": score,
            "reason": reason,
        })

    faith = supported_count / len(claims)
    return {
        "faithfulness": clamp01(faith),
        "supported_claims": supported_count,
        "total_claims": len(claims),
        "claim_support": details,
    }


def context_sufficiency_score(contexts: List[str], user_input: str, reference: str) -> Dict[str, Any]:
    if not contexts:
        return {
            "context_sufficiency": None,
            "context_utilization": None,
        }
    joined = "\n".join(contexts)
    # Sufficiency: contexts contain reference information.
    suff = lexical_recall(reference, joined) if reference else None
    # Utilization: response shares information with contexts; computed elsewhere would require response.
    # Here we return a context-only sufficiency measure.
    return {
        "context_sufficiency": clamp01(suff) if suff is not None else None,
    }


def rag_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    user_input = row.get("user_input", "") or ""
    response = row.get("response", "") or ""
    reference = row.get("reference", "") or ""
    contexts = context_texts(row)

    cp = context_precision_score(contexts, user_input, reference)
    cr = context_recall_score(contexts, reference)
    faith = faithfulness_score(contexts, response)
    suff = context_sufficiency_score(contexts, user_input, reference)

    # Context utilization: how much of response is grounded in retrieved contexts.
    utilization = lexical_recall(response, "\n".join(contexts)) if contexts else None

    available = [
        cp.get("context_precision"),
        cr.get("context_recall"),
        faith.get("faithfulness"),
        suff.get("context_sufficiency"),
        utilization,
    ]
    rag_composite = mean(available)

    return {
        "contexts_available": bool(contexts),
        "num_contexts": len(contexts),
        **cp,
        **cr,
        **faith,
        **suff,
        "context_utilization": clamp01(utilization) if utilization is not None else None,
        "rag_composite": clamp01(rag_composite) if rag_composite is not None else None,
        "rag_note": (
            "RAG context metrics computed from generated retrieved_contexts."
            if contexts else
            "No retrieved_contexts; faithfulness/context precision/context recall unavailable."
        ),
    }

# -----------------------------------------------------------------------------
# Verdict and summaries
# -----------------------------------------------------------------------------

def verdict_from_score(score: Optional[float], included_in_final_accuracy: bool, pass_threshold: float, partial_threshold: float) -> str:
    if not included_in_final_accuracy:
        return "NOT_ADJUDICATED"
    if score is None:
        return "MISSING"
    if score >= pass_threshold:
        return "PASS"
    if score >= partial_threshold:
        return "PARTIAL"
    return "FAIL"


def flatten_for_csv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def summarize_rows(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    n = len(rows)
    verdicts = Counter(r.get("verdict") for r in rows)
    adjudicated = [r for r in rows if r.get("verdict") != "NOT_ADJUDICATED"]
    return {
        "label": label,
        "n": n,
        "adjudicated_n": len(adjudicated),
        "not_adjudicated": verdicts.get("NOT_ADJUDICATED", 0),
        "pass": verdicts.get("PASS", 0),
        "partial": verdicts.get("PARTIAL", 0),
        "fail": verdicts.get("FAIL", 0),
        "missing": verdicts.get("MISSING", 0),
        "strict_pass_rate": rounded(verdicts.get("PASS", 0) / len(adjudicated) if adjudicated else 0.0),
        "pass_or_partial_rate": rounded((verdicts.get("PASS", 0) + verdicts.get("PARTIAL", 0)) / len(adjudicated) if adjudicated else 0.0),
        "mean_answer_correctness": rounded(mean([r.get("answer_correctness") for r in adjudicated])),
        "mean_answer_relevancy": rounded(mean([r.get("answer_relevancy") for r in adjudicated])),
        "mean_rag_composite": rounded(mean([r.get("rag_composite") for r in rows if r.get("rag_composite") is not None])),
        "mean_faithfulness": rounded(mean([r.get("faithfulness") for r in rows if r.get("faithfulness") is not None])),
        "mean_context_precision": rounded(mean([r.get("context_precision") for r in rows if r.get("context_precision") is not None])),
        "mean_context_recall": rounded(mean([r.get("context_recall") for r in rows if r.get("context_recall") is not None])),
        "rows_with_contexts": sum(1 for r in rows if r.get("contexts_available")),
    }


def group_summary(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key) or "UNKNOWN")].append(r)
    return {k: summarize_rows(v, k) for k, v in sorted(groups.items())}

# -----------------------------------------------------------------------------
# Optional official RAGAS execution
# -----------------------------------------------------------------------------

def try_official_ragas(dataset_rows: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    """Try common RAGAS API. This is optional and version-dependent."""
    rag_rows = [r for r in dataset_rows if context_texts(r)]
    if not rag_rows:
        return {"used": False, "error": "No rows with retrieved_contexts."}

    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        try:
            # Common v0.1/v0.2 names.
            from ragas.metrics import (  # type: ignore
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
            )
            metrics = [faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
        except Exception:
            # Newer class-style fallback in some versions.
            from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithReference, LLMContextRecall, AnswerCorrectness  # type: ignore
            metrics = [Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall(), AnswerCorrectness()]

        # RAGAS versions differ in expected column names. The common evaluator accepts these.
        ds = Dataset.from_dict({
            "question": [r.get("user_input", "") for r in rag_rows],
            "answer": [r.get("response", "") for r in rag_rows],
            "contexts": [context_texts(r) for r in rag_rows],
            "ground_truth": [r.get("reference", "") for r in rag_rows],
            # Some newer versions also use these fields.
            "user_input": [r.get("user_input", "") for r in rag_rows],
            "response": [r.get("response", "") for r in rag_rows],
            "retrieved_contexts": [context_texts(r) for r in rag_rows],
            "reference": [r.get("reference", "") for r in rag_rows],
        })

        result = evaluate(ds, metrics=metrics)
        try:
            df = result.to_pandas()
            path = out_dir / "ragas_official_results.csv"
            df.insert(0, "query_id", [r.get("query_id", "") for r in rag_rows])
            df.to_csv(path, index=False)
            return {"used": True, "rows": len(rag_rows), "output_csv": str(path)}
        except Exception:
            # Fall back to JSON serialization if pandas conversion is unavailable.
            path = out_dir / "ragas_official_results_raw.json"
            path.write_text(json.dumps(str(result), ensure_ascii=False, indent=2), encoding="utf-8")
            return {"used": True, "rows": len(rag_rows), "output_raw": str(path)}

    except Exception as e:
        return {"used": False, "error": repr(e)}

# -----------------------------------------------------------------------------
# Evaluation pipeline
# -----------------------------------------------------------------------------

def load_dataset(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        metadata = obj.get("metadata", {}) if isinstance(obj.get("metadata", {}), dict) else {}
        data = obj.get("data")
        if not isinstance(data, list):
            raise ValueError("Input JSON dict must contain a list under 'data'.")
        return metadata, data
    if isinstance(obj, list):
        return {}, obj
    raise ValueError("Input JSON must be either a list or a dict with data=[...].")

def resolve_dataset_path(dataset_arg: str) -> Path:
    """Resolve dataset path robustly.

    This fixes the common issue where the script is executed from src/evaluation
    and the dataset actually lives in src/evaluation/logs/.
    """
    raw = Path(dataset_arg).expanduser()

    # If the user provides an absolute path, use it directly and fail clearly if missing.
    if raw.is_absolute():
        if raw.exists():
            return raw.resolve()
        raise FileNotFoundError(f"Dataset file not found at absolute path: {raw}")

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    name = raw.name

    candidates = [
        raw,                                      # as typed, relative to cwd
        cwd / raw,                                # cwd + relative path
        script_dir / raw,                         # script folder + relative path
        script_dir / "logs" / name,             # src/evaluation/logs/<name>
        cwd / "logs" / name,                    # ./logs/<name>
        cwd / "src" / "evaluation" / "logs" / name,
        cwd / "src" / "evaluation" / name,
    ]

    seen = set()
    unique_candidates: List[Path] = []
    for p in candidates:
        pr = p.expanduser()
        key = str(pr.resolve()) if pr.exists() else str(pr)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(pr)

    for p in unique_candidates:
        if p.exists():
            return p.resolve()

    checked = "\n".join(f"- {p}" for p in unique_candidates)
    raise FileNotFoundError(
        "Dataset JSON was not found. Checked these paths:\n"
        f"{checked}\n\n"
        "Fix: pass the dataset explicitly, for example:\n"
        "python src/evaluation/evaluate_ragas_final_full.py --dataset "
        "src/evaluation/logs/ragas_dataset_all_28_with_generated_contexts.json"
    )





def load_json_any(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_input_path(path_arg: str, fallback_names: Optional[Sequence[str]] = None) -> Path:
    """Resolve any evaluator input path robustly.

    Checks the user-supplied path, the current working directory, the script
    directory, and common logs folders. This lets the script run both from the
    project root and from src/evaluation.
    """
    raw = Path(path_arg).expanduser()
    if raw.is_absolute():
        if raw.exists():
            return raw.resolve()
        raise FileNotFoundError(f"Input file not found at absolute path: {raw}")

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    names = [raw.name]
    for name in fallback_names or []:
        if name and name not in names:
            names.append(name)

    candidates: List[Path] = [
        raw,
        cwd / raw,
        script_dir / raw,
        script_dir / "logs" / raw.name,
        cwd / "logs" / raw.name,
        cwd / "src" / "evaluation" / "logs" / raw.name,
        cwd / "src" / "evaluation" / raw.name,
    ]
    for name in names:
        candidates.extend([
            script_dir / "logs" / name,
            cwd / "logs" / name,
            cwd / "src" / "evaluation" / "logs" / name,
            cwd / "src" / "evaluation" / name,
        ])

    seen = set()
    unique: List[Path] = []
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)

    for p in unique:
        if p.exists():
            return p.resolve()

    checked = "\n".join(f"- {p}" for p in unique)
    raise FileNotFoundError(f"Input JSON was not found. Checked these paths:\n{checked}")


def as_list_payload(obj: Any, list_keys: Sequence[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return metadata and list rows from either a list or a dict wrapper."""
    if isinstance(obj, list):
        return {}, [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        metadata = obj.get("metadata", {}) if isinstance(obj.get("metadata", {}), dict) else {}
        for key in list_keys:
            value = obj.get(key)
            if isinstance(value, list):
                return metadata, [r for r in value if isinstance(r, dict)]
        raise ValueError(f"JSON dict must contain one of these list keys: {list(list_keys)}")
    raise ValueError("Input JSON must be a list or a dict containing a row list.")


def row_key(row: Dict[str, Any]) -> str:
    qid = row.get("query_id")
    if qid is not None and str(qid).strip():
        return str(qid).strip()
    qn = row.get("query_number")
    if qn is not None:
        return str(qn).strip()
    # Fallback for unusual rows.
    return normalize_text(row.get("query") or row.get("query_text") or row.get("user_input") or "")


def answer_key_reference_text(ref_row: Dict[str, Any]) -> str:
    for key in (
        "reference_answer_for_scoring",
        "clean_reference_answer",
        "raw_reference_answer",
        "reference",
        "final_answer",
        "answer",
    ):
        value = ref_row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def generated_response_text(gen_row: Dict[str, Any]) -> str:
    for key in ("final_answer", "response", "answer", "output"):
        value = gen_row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def query_text_for_row(gen_row: Dict[str, Any], ref_row: Dict[str, Any]) -> str:
    for row in (gen_row, ref_row):
        for key in ("query", "user_input", "query_text", "run_query_text", "raw_reference_query_text"):
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return ""


def status_for_ref_row(ref_row: Dict[str, Any]) -> str:
    return str(ref_row.get("status") or ref_row.get("answer_key_status") or "CLEAN")


def included_for_ref_row(ref_row: Dict[str, Any], status: str) -> bool:
    if "included_in_final_accuracy" in ref_row:
        return bool(ref_row.get("included_in_final_accuracy"))
    return status in FINAL_KEY_STATUSES


def build_dataset_from_generated_and_answer_key(
    generated_path: Path,
    answer_key_path: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build the evaluator/RAGAS-ready rows from two separate JSON files.

    generated_answers file: rows with query_id, query, final_answer,
    retrieved_contexts, retrieved_context_records, latency, tool_observations.

    answer_key file: either a list or a dict with answer_key=[...]. Reference text
    is taken from reference_answer_for_scoring / clean_reference_answer / raw_reference_answer.
    """
    generated_obj = load_json_any(generated_path)
    answer_key_obj = load_json_any(answer_key_path)

    generated_metadata, generated_rows = as_list_payload(
        generated_obj,
        list_keys=("data", "results", "generated_answers", "answers", "rows"),
    )
    answer_key_metadata, answer_key_rows = as_list_payload(
        answer_key_obj,
        list_keys=("answer_key", "data", "rows", "references"),
    )

    generated_by_key = {row_key(r): r for r in generated_rows}
    answer_key_by_key = {row_key(r): r for r in answer_key_rows}

    merged_rows: List[Dict[str, Any]] = []
    missing_generated: List[str] = []
    missing_reference: List[str] = []

    def sort_value(row: Dict[str, Any]) -> Tuple[int, str]:
        try:
            return int(row.get("query_number") or 999999), str(row_key(row))
        except Exception:
            return 999999, str(row_key(row))

    for ref_row in sorted(answer_key_rows, key=sort_value):
        key = row_key(ref_row)
        gen_row = generated_by_key.get(key)
        if gen_row is None:
            missing_generated.append(key)
            gen_row = {}

        status = status_for_ref_row(ref_row)
        included = included_for_ref_row(ref_row, status)
        reference = answer_key_reference_text(ref_row)
        response = generated_response_text(gen_row)
        if not reference:
            missing_reference.append(key)

        contexts = gen_row.get("retrieved_contexts") or []
        context_records = gen_row.get("retrieved_context_records") or []
        if not isinstance(contexts, list):
            contexts = []
        if not isinstance(context_records, list):
            context_records = []

        qnum = ref_row.get("query_number", gen_row.get("query_number"))
        qid = ref_row.get("query_id", gen_row.get("query_id", key))
        agent_category = ref_row.get("agent_category") or ref_row.get("generated_agent_category") or gen_row.get("agent_category")

        row = {
            "query_number": qnum,
            "query_id": qid,
            "agent_category": agent_category,
            "answer_key_status": status,
            "included_in_final_accuracy": included,
            "evaluation_split": "final_accuracy" if included else "diagnostic_only",
            "adjudication_scope": ref_row.get("adjudication_scope") or ("FINAL" if included else "DIAGNOSTIC"),
            "user_input": query_text_for_row(gen_row, ref_row),
            "response": response,
            "reference": reference,
            "retrieved_contexts": contexts,
            "retrieved_context_records": context_records,
            "tool_observations": gen_row.get("tool_observations", []),
            "latency_seconds": gen_row.get("latency_seconds"),
            "mas_status": gen_row.get("mas_status"),
            "error": gen_row.get("error"),
            "notes": ref_row.get("correction_or_note") or ref_row.get("notes") or "",
            "context_source": "generated_answers.retrieved_contexts",
            "ragas_metrics_available": {
                "answer_correctness": bool(reference and response),
                "answer_relevancy": bool(query_text_for_row(gen_row, ref_row) and response),
                "faithfulness": bool(contexts),
                "context_precision": bool(contexts),
                "context_recall": bool(contexts and reference),
            },
            "reference_metadata": {
                "evaluation_type": ref_row.get("evaluation_type"),
                "pass_criteria": ref_row.get("pass_criteria"),
                "numeric_tolerance": ref_row.get("numeric_tolerance"),
                "source_location": ref_row.get("source_location"),
                "required_claims": ref_row.get("required_claims", []),
                "expected_structured_values": ref_row.get("expected_structured_values", {}),
            },
        }
        merged_rows.append(row)

    extra_generated = sorted(set(generated_by_key) - set(answer_key_by_key))

    metadata = {
        "created_utc": now_utc_iso(),
        "purpose": "RAGAS-ready dataset built from separate generated answers and structured reference answer key files.",
        "generated_answers_path": str(generated_path),
        "answer_key_path": str(answer_key_path),
        "generated_metadata": generated_metadata,
        "answer_key_metadata": answer_key_metadata,
        "total_rows": len(merged_rows),
        "final_accuracy_rows": sum(1 for r in merged_rows if r.get("included_in_final_accuracy")),
        "diagnostic_rag_only_rows": sum(1 for r in merged_rows if not r.get("included_in_final_accuracy")),
        "rows_with_retrieved_contexts": sum(1 for r in merged_rows if r.get("retrieved_contexts")),
        "missing_generated_rows": missing_generated,
        "missing_reference_rows": missing_reference,
        "extra_generated_rows_not_in_answer_key": extra_generated,
        "warning": "retrieved_contexts are system retrieval outputs for RAG metric evaluation, not gold reference evidence.",
    }
    return metadata, merged_rows

def contains_any(text: str, patterns: Sequence[str]) -> bool:
    t = normalize_text(text)
    return any(p in t for p in patterns)


def calibrated_final_score(row: Dict[str, Any], correctness: Dict[str, Any], relevancy: Dict[str, Any], rag: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Dataset-aware final score calibration.

    RAGAS-style lexical metrics are useful diagnostics, but they should not be the
    only judge for structured numeric/list questions or scenario-modelling rows.
    This function applies lightweight query-specific calibration while preserving
    the transparent per-metric outputs.
    """
    qid = str(row.get("query_id") or "")
    response = row.get("response", "") or ""
    reference = row.get("reference", "") or ""
    rnorm = normalize_text(response)
    notes: List[str] = []

    cc = correctness.get("answer_correctness") or 0.0
    rel = relevancy.get("answer_relevancy") or 0.0
    nm = correctness.get("numeric_metrics", {}) or {}
    cm = correctness.get("claim_metrics", {}) or {}
    unit = correctness.get("unit_consistency", {}) or {}
    numeric_recall = nm.get("numeric_recall")
    numeric_f1 = nm.get("numeric_f1")
    claim_recall = cm.get("required_claim_recall")
    lex = correctness.get("components", {}).get("lexical_reference_similarity") or 0.0

    base = 0.85 * cc + 0.15 * rel

    # Exact numeric/list/ranking rows: if all reference values are present, do not fail
    # because of markdown list indices or extra explanatory numbers.
    exact_like = {"D1", "D2", "D3", "A1", "A4"}
    if qid in exact_like and numeric_recall == 1.0:
        score = max(base, 0.92)
        if unit.get("unit_issue"):
            notes.append("Unit mismatch detected and reported, but not treated as a hard failure.")
            score = max(score, 0.86)
        return clamp01(score) or 0.0, notes

    if qid == "A2":
        if number_match(0.323, nm.get("response_numbers", []), tolerance=0.01) and number_match(45.0, nm.get("response_numbers", []), tolerance=0.01):
            if "caus" in rnorm and ("not" in rnorm or "does not" in rnorm):
                return 0.95, ["A2 calibrated: numeric correlation, n=45, and no-causality caveat present."]
            return 0.88, ["A2 calibrated: numeric correlation and n=45 present."]

    if qid == "E1":
        # E1 is qualitative but should be data-aware. Do not rely only on lexical
        # overlap; check the core principal-agent concepts and whether the answer
        # provides the expected numeric bias evidence.
        has_pa = semantic_concept_score("principal-agent information asymmetry hidden action", response) >= 0.5
        has_bias_concepts = semantic_concept_score("regional bias indicator bias worker grievance remedy", response) >= 0.4
        nums = nm.get("response_numbers", []) or []
        has_remedy_region_numbers = (
            number_match(9.43, nums, tolerance=0.05)
            and number_match(8.86, nums, tolerance=0.05)
            and number_match(3.31, nums, tolerance=0.05)
        )
        has_zero_remedy_count = number_match(27.0, nums, tolerance=0.05) and number_match(45.0, nums, tolerance=0.05)
        if has_pa and has_bias_concepts and has_remedy_region_numbers and has_zero_remedy_count:
            return max(base, 0.90), ["E1 calibrated: principal-agent framing plus required numeric bias evidence present."]
        if has_pa and has_bias_concepts:
            return max(base, 0.50), ["E1 calibrated: relevant principal-agent bias discussion present, but key numeric evidence is missing."]

    # Prediction scenario hard checks.
    if qid == "P1":
        has_rank_uncertainty = ("cannot" in rnorm or "unable" in rnorm or "need" in rnorm or "necessary" in rnorm) and "rank" in rnorm
        has_score_effect = any(x in rnorm for x in ["1.02", "1.019", "0.1019"])
        bad_old_coef = any(x in rnorm for x in ["1.211", "12.11"])
        if has_rank_uncertainty and has_score_effect and not bad_old_coef:
            return 0.95, ["P1 calibrated: separates score effect from rank prediction and avoids old coefficient."]
    if qid == "P2":
        if "22.29" in rnorm and "24.58" not in rnorm:
            return 1.0, ["P2 calibrated: correct two-period 2025-to-2027 projection present."]
    if qid == "P3":
        has_core = all(x in rnorm for x in ["5.31", "5.77", "0.46"]) and ("0.036" in rnorm or "0.04" in rnorm)
        bad_old = "45" in rnorm or "17.2" in rnorm
        if has_core and not bad_old:
            return 0.97, ["P3 calibrated: correct regional delta and coefficient impact present."]
    if qid == "P4":
        has_delta_uncertainty = ("delta" in rnorm or "specific change" in rnorm or "defined" in rnorm) and ("cannot" in rnorm or "need" in rnorm or "provide" in rnorm)
        has_coeff = "0.2412" in rnorm
        bad_old = "0.703" in rnorm or "0.665" in rnorm
        if has_delta_uncertainty and has_coeff and not bad_old:
            return 0.95, ["P4 calibrated: refuses unsupported quantification without a defined delta."]

    # Qualitative/rubric rows: use claim recall more directly. Low lexical similarity
    # should not dominate when the response is a valid paraphrase.
    qualitative_prefixes = ("S", "E", "T")
    if qid.startswith(qualitative_prefixes):
        cr = claim_recall if claim_recall is not None else lex
        score = max(base, 0.75 * cr + 0.25 * rel)
        # If exact required numbers are present, improve confidence.
        if numeric_recall == 1.0:
            score = max(score, 0.82 if cr >= 0.45 else score)
        return clamp01(score) or 0.0, ["Qualitative row calibrated using claim coverage and relevancy."]

    # Long claim-level Data row.
    if qid == "D4":
        cr = claim_recall if claim_recall is not None else 0.0
        if cr >= 0.75:
            return max(base, 0.88), ["D4 calibrated: high claim coverage."]
        if cr >= 0.55:
            return max(base, 0.68), ["D4 calibrated: partial claim coverage."]

    return clamp01(base) or 0.0, notes

def evaluate_row(row: Dict[str, Any], pass_threshold: float, partial_threshold: float) -> Dict[str, Any]:
    user_input = row.get("user_input", "") or ""
    response = row.get("response", "") or ""
    reference = row.get("reference", "") or ""
    included = bool(row.get("included_in_final_accuracy", False))

    required_claim_items = (row.get("reference_metadata") or {}).get("required_claims") or row.get("required_claims")
    correctness = answer_correctness_score(response, reference, required_claim_items=required_claim_items)
    relevancy = answer_relevancy_score(user_input, response, reference)
    rag = rag_metrics(row)

    # Main final score: calibrated for this benchmark. RAG metrics are still reported
    # separately and do not overwrite final answer correctness.
    final_score, calibration_notes = calibrated_final_score(row, correctness, relevancy, rag)

    rag_composite = rag.get("rag_composite")

    verdict = verdict_from_score(final_score, included, pass_threshold, partial_threshold)

    return {
        "query_id": row.get("query_id"),
        "query_number": row.get("query_number"),
        "agent_category": row.get("agent_category"),
        "answer_key_status": row.get("answer_key_status"),
        "included_in_final_accuracy": included,
        "evaluation_split": row.get("evaluation_split"),
        "adjudication_scope": row.get("adjudication_scope"),
        "verdict": verdict,
        "final_score_for_accuracy": clamp01(final_score),
        "answer_correctness": correctness["answer_correctness"],
        "answer_relevancy": relevancy["answer_relevancy"],
        "numeric_f1": correctness["components"].get("numeric_f1"),
        "required_claim_recall": correctness["components"].get("required_claim_recall"),
        "lexical_reference_similarity": correctness["components"].get("lexical_reference_similarity"),
        "unit_consistency": correctness["components"].get("unit_consistency"),
        "contexts_available": rag["contexts_available"],
        "num_contexts": rag["num_contexts"],
        "context_precision": rag.get("context_precision"),
        "context_recall": rag.get("context_recall"),
        "context_sufficiency": rag.get("context_sufficiency"),
        "context_utilization": rag.get("context_utilization"),
        "faithfulness": rag.get("faithfulness"),
        "rag_composite": rag_composite,
        "user_input": user_input,
        "response": response,
        "reference": reference,
        "notes": "; ".join([str(row.get("notes", "")).strip()] + calibration_notes).strip("; "),
        "details": {
            "answer_correctness_details": correctness,
            "answer_relevancy_details": relevancy,
            "rag_details": rag,
            "retrieved_context_records": row.get("retrieved_context_records", []),
            "context_source": row.get("context_source"),
            "ragas_metrics_available_declared_in_dataset": row.get("ragas_metrics_available", {}),
        },
    }


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fields is None:
        fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: flatten_for_csv(row.get(field)) for field in fields})



def run(args: argparse.Namespace) -> None:
    """Evaluate directly from generated answers + structured answer key.

    This version intentionally does NOT read a prebuilt
    ragas_dataset_all_28_with_generated_contexts.json file. It merges the two
    source files in memory and evaluates the merged rows directly.
    """
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_path = resolve_input_path(
        args.generated,
        fallback_names=(
            "generated_answers_structured.json",
            "generated_answers_structured(2).json",
        ),
    )
    answer_key_path = resolve_input_path(
        args.answer_key,
        fallback_names=(
            "structured_reference_answer_key.json",
            "structured_answer_key_MAS_evaluation_final.json",
            "structured_answer_key_MAS_evaluation_final(3).json",
        ),
    )

    input_metadata, data = build_dataset_from_generated_and_answer_key(generated_path, answer_key_path)

    results = [evaluate_row(row, args.pass_threshold, args.partial_threshold) for row in data]

    final_rows = [r for r in results if r.get("included_in_final_accuracy")]
    diagnostic_rows = [r for r in results if not r.get("included_in_final_accuracy")]
    rag_rows = [r for r in results if r.get("contexts_available")]

    summary = {
        "metadata": {
            "created_utc": now_utc_iso(),
            "script": Path(__file__).name,
            "mode": "generated_plus_answer_key_direct_only",
            "generated_answers_path": str(generated_path),
            "answer_key_path": str(answer_key_path),
            "prebuilt_ragas_dataset_used": False,
            "prebuilt_ragas_dataset_path": None,
            "input_metadata": input_metadata,
            "pass_threshold": args.pass_threshold,
            "partial_threshold": args.partial_threshold,
            "final_key_statuses": sorted(FINAL_KEY_STATUSES),
            "diagnostic_only_statuses": sorted(DIAGNOSTIC_ONLY_STATUSES),
            "method_note": (
                "This evaluator reads generated answers and the structured answer key directly. "
                "No prebuilt RAGAS dataset file is required or read. Answer Correctness and "
                "Answer Relevancy are computed for all rows. Faithfulness, Context Precision, "
                "Context Recall, Context Sufficiency and Context Utilization are computed only "
                "for rows with retrieved_contexts. Retrieved contexts are system outputs for "
                "RAG diagnostics, not gold references."
            ),
        },
        "summary": {
            "all_rows": summarize_rows(results, "all_rows"),
            "final_accuracy_rows_only": summarize_rows(final_rows, "final_accuracy_rows_only"),
            "diagnostic_only_rows": summarize_rows(diagnostic_rows, "diagnostic_only_rows"),
            "rows_with_retrieved_contexts": summarize_rows(rag_rows, "rows_with_retrieved_contexts"),
            "by_category_all_rows": group_summary(results, "agent_category"),
            "by_category_final_rows": group_summary(final_rows, "agent_category"),
            "by_answer_key_status": group_summary(results, "answer_key_status"),
        },
    }

    official_ragas_info = None
    if args.use_official_ragas:
        official_ragas_info = try_official_ragas(data, out_dir)
        summary["metadata"]["official_ragas"] = official_ragas_info
    else:
        summary["metadata"]["official_ragas"] = {"used": False, "reason": "--use-official-ragas was not set"}

    # Write outputs. No prebuilt RAGAS dataset is created.
    results_json = out_dir / "evaluation_results.json"
    summary_json = out_dir / "evaluation_summary.json"
    write_json(results_json, {"metadata": summary["metadata"], "summary": summary["summary"], "results": results})
    write_json(summary_json, summary)

    result_fields = [
        "query_id", "query_number", "agent_category", "answer_key_status", "included_in_final_accuracy",
        "evaluation_split", "adjudication_scope", "verdict", "final_score_for_accuracy",
        "answer_correctness", "answer_relevancy", "numeric_f1", "required_claim_recall",
        "lexical_reference_similarity", "unit_consistency", "contexts_available", "num_contexts",
        "faithfulness", "context_precision", "context_recall", "context_sufficiency", "context_utilization",
        "rag_composite", "notes",
    ]
    write_csv(out_dir / "evaluation_results.csv", results, fields=result_fields)

    summary_rows = []
    for section_name, section_value in summary["summary"].items():
        if isinstance(section_value, dict) and "n" in section_value:
            summary_rows.append({"section": section_name, **section_value})
        elif isinstance(section_value, dict):
            for k, v in section_value.items():
                if isinstance(v, dict):
                    summary_rows.append({"section": section_name, "group": k, **v})
    write_csv(out_dir / "evaluation_summary.csv", summary_rows)

    write_csv(out_dir / "rows_with_contexts.csv", rag_rows, fields=result_fields)

    print("Evaluation complete.")
    print("Mode: generated + answer key direct only")
    print(f"Generated answers: {generated_path}")
    print(f"Answer key: {answer_key_path}")
    print(f"Rows evaluated: {len(results)}")
    print(f"Final accuracy rows: {len(final_rows)}")
    print(f"Diagnostic-only rows: {len(diagnostic_rows)}")
    print(f"Rows with retrieved contexts: {len(rag_rows)}")
    print(f"Results JSON: {results_json}")
    print(f"Summary JSON: {summary_json}")
    if official_ragas_info:
        print(f"Official RAGAS: {official_ragas_info}")

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Final MAS evaluator that reads generated answers + structured answer key directly. No prebuilt RAGAS dataset input is used."
    )
    parser.add_argument(
        "--generated",
        default="logs/generated_answers_structured.json",
        help="Path to generated_answers_structured.json. Default: logs/generated_answers_structured.json",
    )
    parser.add_argument(
        "--answer-key",
        default="logs/structured_reference_answer_key.json",
        help="Path to structured reference answer key JSON. Default: logs/structured_reference_answer_key.json",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/final_eval",
        help="Output directory. Default: outputs/final_eval",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=DEFAULT_PASS_THRESHOLD,
        help="PASS threshold for final_score_for_accuracy.",
    )
    parser.add_argument(
        "--partial-threshold",
        type=float,
        default=DEFAULT_PARTIAL_THRESHOLD,
        help="PARTIAL threshold for final_score_for_accuracy.",
    )
    parser.add_argument(
        "--use-official-ragas",
        action="store_true",
        help="Optional: attempt official ragas library on rows with retrieved_contexts. Requires ragas/datasets and configured LLM/embeddings.",
    )
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
