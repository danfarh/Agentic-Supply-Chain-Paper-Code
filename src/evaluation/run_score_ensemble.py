"""Equal-weight fusion of company scores from the Data and Document tools."""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

# Exact workbook names and scorecard filename suffixes (not score values).
SOURCES = {
    "Samsung Electronics Co. Ltd.": "Samsung",
    "LG Electronics Inc.": "LG Electronics",
    "SK Hynix Inc.": "SK Hynix Inc",
    "Sony Corp.": "Sony",
    "Hon Hai Precision Industry Co. Ltd. (Foxconn)": "Hon Hai",
    "Taiwan Semiconductor Manufacturing Co. Ltd.": "TSMC",
    "Canon Inc.": "Canon",
    "Panasonic Corp.": "Panasonic",
    "Murata Manufacturing Co. Ltd.": "Murata Manufacturing",
    "Fujifilm Holdings Corp.": "Fujifilm",
    "Keyence Corp.": "Keyence",
    "Kyocera Corp.": "Kyocera",
    "Xiaomi Corp.": "Xiaomi",
    "Luxshare Precision Industry Co. Ltd.": "Luxshare Precision",
    "Semiconductor Manufacturing International Corp": "SMIC",
    "BOE Technology Group Co. Ltd.": "BOE Technology",
    "Hewlett Packard Enterprise Co. (HPE)": "HPE",
    "Cisco Systems Inc.": "Cisco",
    "HP Inc.": "HP",
    "Apple Inc.": "Apple",
}
SCORE = re.compile(r"\bOVERALL\s+SCORE\s*[:\-]?\s*(\d+(?:\.\d+)?)\s+out\s+of\s+100\b", re.I)


def company_scores(text):
    pairs = {}
    for name, value in re.findall(r"^(?:-\s+|\d+\.\s+)(.+?):\s*(\d+(?:\.\d+)?)\s*$",
                                  text.replace("**", ""), re.M):
        if name in pairs:
            raise ValueError(f"Duplicate score for {name}")
        pairs[name] = float(value)
    return pairs


def pdf_score(company, live):
    suffix = SOURCES[company]
    filename = f"2025 KTC ICT Scorecard_{suffix}.pdf"
    path = ROOT / "data" / "pdfs" / filename
    if not path.exists():
        return None, filename, "PDF unavailable"
    if live:
        from src.tools.document_tools import company_pdf_rag
        payload = json.loads(company_pdf_rag.invoke(dict(
            company_name=path.stem.split(" ", 1)[1],
            question=f"What is the OVERALL SCORE out of 100 on the 2025 ICT scorecard for {company}?",
            k=5)))
        if payload.get("error"):
            raise RuntimeError(f"{company}: {payload['error']}")
        passages = [r["content"] for r in payload["retrieved_contexts"]
                    if r.get("source", "").replace(chr(92), "/").split("/")[-1] == filename
                    and r.get("page") == 1]
    else:
        from pypdf import PdfReader
        passages = [PdfReader(path).pages[0].extract_text() or ""]
    scores = {float(s) for p in passages for s in SCORE.findall(p)}
    if len(scores) == 1 and 0 <= min(scores) <= 100:
        return scores.pop(), filename, "matched page 1"
    return None, filename, "no unique labelled score"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(live):
    logs = HERE / "logs"
    generated = json.loads((logs / "generated_answers_structured.json").read_text(encoding="utf-8-sig"))
    key = json.loads((logs / "structured_reference_answer_key.json").read_text(encoding="utf-8-sig"))
    answers = {r["query_id"]: r for r in generated}
    references = {r["query_id"]: r for r in key["answer_key"]}
    companies = {}
    for q, tool in [("D1", "list_companies_by_region"), ("D3", "get_top_companies_by_metric")]:
        row = answers[q]
        observations = [o for o in row["tool_observations"] if o["tool_name"] == tool]
        if len(observations) != 1 or row.get("error"):
            raise ValueError(f"Check the original log for {q}")
        data = company_scores(observations[0]["observation"])
        mas = company_scores(row["final_answer"])
        refs = company_scores(references[q]["reference_answer_for_scoring"])
        if len(refs) != {"D1": 16, "D3": 5}[q]:
            raise ValueError(f"Unexpected reference format for {q}")
        for company, ref in refs.items():
            values = dict(reference=ref, data=data.get(company), mas=mas.get(company))
            if company in companies:
                if any(companies[company][k] != v for k, v in values.items()):
                    raise ValueError(f"Inconsistent repeated observation: {company}")
                companies[company]["queries"] += "+" + q
            else:
                companies[company] = dict(company=company, queries=q, **values)

    rows = list(companies.values())
    for row in rows:
        pdf, source, status = pdf_score(row["company"], live)
        row.update(pdf=pdf, pdf_source=source, pdf_status=status, ensemble=None)
        if pdf is not None and row["data"] is not None:
            row["ensemble"] = (row["data"] + pdf) / 2
    methods = ["data", "pdf", "ensemble", "mas"]
    common = [r for r in rows if all(r[m] is not None for m in methods)]
    summary = []
    for m in methods:
        errors = [abs(r[m] - r["reference"]) for r in common]
        summary.append(dict(method=m, total=len(rows), available=sum(r[m] is not None for r in rows),
                            common_n=len(common), common_mae=mean(errors) if errors else None))
    out = HERE / "outputs" / "score_ensemble" / ("rag" if live else "page_lookup")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "company_scores.csv", rows)
    write_csv(out / "method_comparison.csv", summary)
    for item in summary:
        print(item)
    paired = [r for r in rows if r["ensemble"] is not None]
    print("Disagreements / paired:", sum(r["data"] != r["pdf"] for r in paired), "/", len(paired))
    print("Saved to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-lookup", action="store_true", help="Offline first-page extraction baseline")
    main(live=not parser.parse_args().page_lookup)
