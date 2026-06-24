"""Benchmark queries for the MAS ablation study.

Keep query IDs aligned with the structured answer key and evaluator.
"""
from __future__ import annotations

from typing import Any, Dict, List

TEST_CASES: List[Dict[str, Any]] = [
    # 1. Data Agent
    {"query_number": 1, "query_id": "D1", "agent_category": "Data Agent", "query": "Filter the companies in the Scoring sheet by Region = `Asia` and list their Company Names and Total Benchmark Scores."},
    {"query_number": 2, "query_id": "D2", "agent_category": "Data Agent", "query": "From the Non-Scored Research sheet, extract all companies that disclose sourcing from `China` and list their Company Names and Market Caps."},
    {"query_number": 3, "query_id": "D3", "agent_category": "Data Agent", "query": "Clean the Scoring sheet by removing any rows with missing or zero Total Benchmark Scores, then list the top 5 companies by Total Benchmark Score descending."},
    {"query_number": 4, "query_id": "D4", "agent_category": "Data Agent", "query": "Extract from the Detailed Scoring & Research sheet the comment for Amazon.com Inc. on indicator 1.1 (Supplier Code of Conduct)."},

    # 2. Analysis Agent
    {"query_number": 5, "query_id": "A1", "agent_category": "Analysis Agent", "query": "Compute the average Total Benchmark Score across all companies in the Scoring sheet, and the standard deviation."},
    {"query_number": 6, "query_id": "A2", "agent_category": "Analysis Agent", "query": "Calculate the correlation between Market Cap and Total Benchmark Score using the Scoring sheet data."},
    {"query_number": 7, "query_id": "A3", "agent_category": "Analysis Agent", "query": "Perform k-means clustering (k=3) on companies based on Total Benchmark Score and Purchasing Practices score from Scoring sheet, and list clusters."},
    {"query_number": 8, "query_id": "A4", "agent_category": "Analysis Agent", "query": "Compute the median score for each theme (e.g., Commitment & Governance) across North American companies."},

    # 3. Research Agent
    {"query_number": 9, "query_id": "R1", "agent_category": "Research Agent", "query": "Cross-verify the high-risk sourcing countries for Amazon from the PDF with current ILO statistics on forced labour prevalence in China and Malaysia."},
    {"query_number": 10, "query_id": "R2", "agent_category": "Research Agent", "query": "Search for the latest ILO report on forced labour in the ICT sector and compare it to the average KTC Remedy score (7/100)."},
    {"query_number": 11, "query_id": "R3", "agent_category": "Research Agent", "query": "Validate Samsung's top rank in the KTC Total Benchmark dataset by searching for recent news on its supply chain practices."},
    {"query_number": 12, "query_id": "R4", "agent_category": "Research Agent", "query": "Fetch external data on global average market cap for ICT semiconductors and compare to KTC dataset average."},

    # 4. Prediction Agent
    {"query_number": 13, "query_id": "P1", "agent_category": "Prediction Agent", "query": "Based on historical ranks (e.g., Amazon 2022 rank 8, 2025 rank 10), predict Amazon's 2027 rank if it improves Remedy by 10 points."},
    {"query_number": 14, "query_id": "P2", "agent_category": "Prediction Agent", "query": "Project the industry average Total Benchmark Score for 2027 if scores increase by 5% annually."},
    {"query_number": 15, "query_id": "P3", "agent_category": "Prediction Agent", "query": "Model score improvement for Asian companies if they match North America's average Purchasing Practices score. (5.77)"},
    {"query_number": 16, "query_id": "P4", "agent_category": "Prediction Agent", "query": "Predict the impact on Apple's score if it addresses Uyghur forced labour allegations from data."},

    # 5. Synthesis Agent
    {"query_number": 17, "query_id": "S1", "agent_category": "Synthesis Agent", "query": "Synthesise key insights on top 5 companies' strengths and weaknesses from Scoring sheet."},
    {"query_number": 18, "query_id": "S2", "agent_category": "Synthesis Agent", "query": "Compile a report on regional differences in Remedy scores, with ethical note on data biases."},
    {"query_number": 19, "query_id": "S3", "agent_category": "Synthesis Agent", "query": "Synthesise opportunities for improvement from Amazon PDF."},
    {"query_number": 20, "query_id": "S4", "agent_category": "Synthesis Agent", "query": "Compile insights on correlations between Market Cap and scores, noting ethical implications."},

    # 6. Ethics Agent
    {"query_number": 21, "query_id": "E1", "agent_category": "Ethics Agent", "query": "Evaluate potential biases in the Scoring sheet data, using principal-agent theory."},
    {"query_number": 22, "query_id": "E2", "agent_category": "Ethics Agent", "query": "Assess if low Remedy scores (avg 7) could amplify ethical risks in supply chains."},
    {"query_number": 23, "query_id": "E3", "agent_category": "Ethics Agent", "query": "Check for regional bias in Non-Scored Research (e.g., more \"Yes\" for UK MSA in NA/Europe)."},
    {"query_number": 24, "query_id": "E4", "agent_category": "Ethics Agent", "query": "Evaluate ethical risks in predicting improvements for low-scorers like BOE (0 score)."},

    # 7. Text Mining Agent
    {"query_number": 25, "query_id": "T1", "agent_category": "Text Mining Agent", "query": "Extract and summarise key phrases from Amazon PDF on `Opportunities for Improvement`."},
    {"query_number": 26, "query_id": "T2", "agent_category": "Text Mining Agent", "query": "Perform sentiment analysis on the Detailed Scoring & Research comments for Samsung."},
    {"query_number": 27, "query_id": "T3", "agent_category": "Text Mining Agent", "query": "Mine the Amazon PDF for mentions of `forced labour` and categorise themes."},
    {"query_number": 28, "query_id": "T4", "agent_category": "Text Mining Agent", "query": "Analyse text from Non-Scored Research on UK MSA compliance and identify patterns."},
]
