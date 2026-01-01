import pandas as pd
from langchain_core.tools import tool

from src.data_loader import get_global_context


@tool
def list_companies_by_region(region_substring: str) -> str:
    """
    DATA AGENT:
    List companies in the Scoring sheet whose Region contains the given substring
    (case-insensitive), along with their Total_Benchmark scores, sorted descending.

    Parameters:
    - region_substring: e.g. "Asia", "North America", "Europe"
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    required = ["Region", "Company", "Total_Benchmark"]
    if not all(c in scoring.columns for c in required):
        return "ERROR: Required columns (Region, Company, Total_Benchmark) not found in scoring sheet."

    mask = scoring["Region"].astype(str).str.contains(region_substring, case=False, na=False)
    subset = scoring[mask]
    if subset.empty:
        return f"No companies found for Region containing '{region_substring}'."

    subset = subset.sort_values("Total_Benchmark", ascending=False)

    lines = [
        f"Companies with Region containing '{region_substring}' (sorted by Total Benchmark):"
    ]
    for _, row in subset.iterrows():
        lines.append(f"- {row['Company']}: {row['Total_Benchmark']:.2f}")

    return "\n".join(lines)


@tool
def list_sourcing_companies(country_name: str) -> str:
    """
    DATA AGENT (GENERALIZED):
    From the Non-Scored Research sheet, list all companies that disclose sourcing from
    the given country_name (e.g., "China", "Malaysia", "Vietnam") and show their Market Caps.

    Parameters:
    - country_name: case-insensitive substring used to detect sourcing columns and text.
    """
    ctx = get_global_context()
    non_scored = ctx.non_scored
    if non_scored is None:
        return "ERROR: Non-Scored Research sheet is not loaded."

    df = non_scored
    cn = country_name.strip().lower()
    country_cols = [c for c in df.columns if cn in str(c).strip().lower()]

    if country_cols:
        col = country_cols[0]
        mask = df[col].astype(str).str.lower().isin(["yes", "y", "true", "1"])
    else:
        mask = df.apply(
            lambda row: row.astype(str).str.contains(country_name, case=False, na=False).any(),
            axis=1
        )

    subset = df[mask]
    if subset.empty:
        return f"No companies disclosing sourcing from '{country_name}' were found in Non-Scored Research."

    lines = [f"Companies that disclose sourcing from '{country_name}' (with Market Caps where available):"]
    for _, row in subset.iterrows():
        company = row.get("Company", "Unknown company")
        mc = row.get("Market_Cap", None)
        if pd.isna(mc):
            lines.append(f"- {company}: Market Cap = N/A")
        else:
            lines.append(f"- {company}: Market Cap = {mc}")
    return "\n".join(lines)


@tool
def top_companies_by_total_benchmark(n: int = 5, drop_zero: bool = True) -> str:
    """
    DATA AGENT:
    Clean the Scoring sheet (remove missing Total_Benchmark and optionally zero scores),
    and list the top N companies by Total_Benchmark descending.

    Parameters:
    - n: number of top companies (default 5)
    - drop_zero: if True, remove rows with Total_Benchmark <= 0
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    if not all(c in scoring.columns for c in ["Company", "Total_Benchmark"]):
        return "ERROR: Required columns (Company, Total_Benchmark) not found."

    df = scoring.dropna(subset=["Total_Benchmark"]).copy()
    df["Total_Benchmark"] = df["Total_Benchmark"].astype(float)
    if drop_zero:
        df = df[df["Total_Benchmark"] > 0]

    df = df.sort_values("Total_Benchmark", ascending=False).head(n)
    if df.empty:
        return "No companies remain after cleaning by Total Benchmark."

    lines = [f"Top {len(df)} companies by Total Benchmark:"]
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        lines.append(f"{i}. {row['Company']}: {row['Total_Benchmark']:.2f}")
    return "\n".join(lines)


@tool
def get_indicator_comment(company_keyword: str, indicator_code: str) -> str:
    """
    DATA AGENT TOOL:
    - If `indicator_code` is specified (e.g., "1.1"), it finds rows from the 'Detailed Scoring & Research'
      sheet that match that company and returns the comments.
    - If `indicator_code` is empty, it retrieves all comments for that company (useful for sentiment analysis).
    """
    # 1. Access the global context
    ctx = get_global_context()

    detailed = ctx.detailed
    if detailed is None:
        return "ERROR: Detailed scoring sheet is not loaded."

    # Copy and flatten columns if MultiIndex
    df = detailed.copy()
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for i, col in enumerate(df.columns):
            parts = [str(x) for x in col if pd.notna(x) and str(x).strip() != ""]
            name = " ".join(parts).strip()
            if not name:
                name = f"col_{i}"
            new_cols.append(name)
        df.columns = new_cols

    # Normalise column names
    df.columns = [str(c).strip() for c in df.columns]

    # 2. Find company column
    company_col = None
    preferred_company_keywords = ["company", "issuer", "name"]
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in preferred_company_keywords):
            company_col = c
            break

    if company_col is None:
        obj_cols = [c for c in df.columns if pd.api.types.is_object_dtype(df[c])]
        if obj_cols:
            company_col = obj_cols[0]
        else:
            return "ERROR: Could not locate a company/name column in the Detailed scoring sheet."

    # 3. Filter by company keyword
    mask_company = df[company_col].astype(str).str.contains(company_keyword, case=False, na=False)

    if not mask_company.any():
        any_mask = pd.DataFrame(
            {c: df[c].astype(str).str.contains(company_keyword, case=False, na=False) for c in df.columns}
        ).any(axis=1)
        mask_company = any_mask

    filtered = df[mask_company]
    if filtered.empty:
        return f"No rows found for company containing '{company_keyword}'."

    # 4. Find indicator column
    indicator_col = None
    for c in df.columns:
        cl = c.lower()
        if "indicator" in cl or "code" in cl or "id" in cl:
            indicator_col = c
            break

    # If indicator_code is provided, filter by it
    indicator_code = (indicator_code or "").strip()
    if indicator_code and indicator_col is not None:
        mask_ind = filtered[indicator_col].astype(str).str.contains(indicator_code, na=False)
        filtered = filtered[mask_ind]
        if filtered.empty:
            return f"No rows found for company '{company_keyword}' with indicator '{indicator_code}'."

    # 5. Identify comment-like columns
    comment_keywords = ("comment", "rationale", "note", "explanation", "research", "detail", "summary")
    comment_cols = [
        c for c in df.columns
        if c != company_col
           and (indicator_col is None or c != indicator_col)
           and any(k in c.lower() for k in comment_keywords)
    ]

    if not comment_cols:
        comment_cols = [
            c for c in df.columns
            if c not in (company_col, indicator_col)
               and pd.api.types.is_object_dtype(df[c])
        ]

    if not comment_cols:
        return "ERROR: Could not identify any comment-like columns."

    # 6. Build text
    rows_text = []
    for _, row in filtered.iterrows():
        parts = []
        if indicator_col is not None:
            ind_val = str(row[indicator_col]).strip()
            if ind_val and ind_val.lower() != "nan":
                parts.append(f"[{ind_val}]")
        for c in comment_cols:
            val = str(row[c]).strip()
            if val and val.lower() != "nan":
                parts.append(val)
        if parts:
            rows_text.append(" ".join(parts))

    if not rows_text:
        return "No non-empty comments found."

    return "\n\n".join(rows_text)
