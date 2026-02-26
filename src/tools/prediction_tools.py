from __future__ import annotations

from typing import Optional, List, Tuple

import pandas as pd
from langchain_core.tools import tool
from sklearn.linear_model import LinearRegression

from src.data_loader import get_global_context


def _normalize_text(x: str) -> str:
    return str(x or "").strip().lower()


def _pick_company_rows(
        scoring: pd.DataFrame,
        company_keyword: str,
        *,
        allow_multiple: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Returns (matches_df, note).
    - Prefer exact match (case-insensitive).
    - Else fallback to contains match.
    - If multiple matches and allow_multiple=False -> return empty with note.
    """
    if "Company" not in scoring.columns:
        return pd.DataFrame(), "ERROR: Company column not found."

    kw = _normalize_text(company_keyword)
    if not kw:
        return pd.DataFrame(), ""

    companies = scoring["Company"].astype(str)

    # 1) Exact match (case-insensitive)
    exact_mask = companies.map(_normalize_text) == kw
    exact_df = scoring.loc[exact_mask].copy()
    if not exact_df.empty:
        return exact_df, "Matched by exact company name (case-insensitive)."

    # 2) Contains match (case-insensitive)
    contains_mask = companies.str.contains(company_keyword, case=False, na=False)
    contains_df = scoring.loc[contains_mask].copy()

    if contains_df.empty:
        return pd.DataFrame(), f"No company found matching '{company_keyword}'."

    if (not allow_multiple) and (len(contains_df) > 1):
        sample = ", ".join(contains_df["Company"].astype(str).head(5).tolist())
        return (
            pd.DataFrame(),
            f"ERROR: '{company_keyword}' matches {len(contains_df)} companies "
            f"(e.g., {sample}). Please provide a more specific keyword or full name."
        )

    return contains_df, "Matched by substring (case-insensitive)."


@tool
def project_metric_growth(
    column_name: str, 
    annual_growth_rate: float, 
    current_year: int = 2025, 
    target_year: int = 2027
) -> str:
    """
    PREDICTION AGENT:
    Project the future industry average for ANY numeric metric, assuming a compound annual growth rate.
    
    Parameters:
    - column_name: The metric to project (e.g., 'Total_Benchmark', 'Purchasing Practices').
    - annual_growth_rate: The expected annual growth as a decimal (e.g., 0.05 for 5%).
    - current_year: The base year (default is 2025).
    - target_year: The future year (default is 2027).
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet not loaded."

    # Find the exact column dynamically
    col_match = next((c for c in scoring.columns if column_name.lower() in str(c).lower()), None)
    if not col_match:
        return f"ERROR: Column '{column_name}' not found."

    # Convert to numeric and drop empty rows
    vals = pd.to_numeric(scoring[col_match], errors="coerce").dropna()
    if vals.empty:
        return f"ERROR: No valid numeric data for '{col_match}'."

    current_avg = float(vals.mean())
    years_diff = target_year - current_year
    
    if years_diff < 0:
        return "ERROR: target_year must be >= current_year."

    # Compound growth formula: Future Value (FV) = Present Value (PV) * (1 + r)^n
    projected_avg = current_avg * ((1 + annual_growth_rate) ** years_diff)

    return (
        f"Projection for '{col_match}':\n"
        f"- Current Average ({current_year}): {current_avg:.2f}\n"
        f"- Projected Average ({target_year}): {projected_avg:.2f} "
        f"(assuming {annual_growth_rate*100:.1f}% annual growth over {years_diff} years)."
    )


@tool
def model_improvement_impact(
        region_name: str,
        indicator_name: str,
        target_score: Optional[float] = None,
        target_region: Optional[str] = None
) -> str:
    """
    PREDICTION AGENT (GENERALIZED):
    Calculate how much a specific Region would need to improve a specific indicator
    to reach a target.

    You must provide EITHER:
    1. target_score: A specific numeric score (e.g., 50.0).
    2. target_region: A region name (e.g., "North America") to match its average.

    Parameters:
    - region_name: e.g. "Asia", "Europe" (the region improving its score).
    - indicator_name: e.g. "Purchasing_Practices", "Recruitment".
    - target_score: (Optional) specific target number.
    - target_region: (Optional) name of region whose average should be the target.
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    # Validation
    if "Region" not in scoring.columns:
        return "ERROR: Region column missing."
    if indicator_name not in scoring.columns:
        return f"ERROR: Indicator column '{indicator_name}' not found."
    if target_score is None and target_region is None:
        return "ERROR: You must provide either 'target_score' or 'target_region'."

    # 1. Get stats for the Subject Region
    region_mask = scoring["Region"].astype(str).str.contains(region_name, case=False, na=False)
    region_df = scoring[region_mask]
    if region_df.empty:
        return f"No companies found in region containing '{region_name}'."

    indicator_vals = pd.to_numeric(region_df[indicator_name], errors="coerce").dropna()
    if indicator_vals.empty:
        return f"No numeric data for indicator '{indicator_name}' in region '{region_name}'."

    current_mean = float(indicator_vals.mean())

    # 2. Determine the Target Score
    final_target = 0.0
    target_desc = ""

    if target_score is not None:
        final_target = target_score
        target_desc = f"fixed target of {target_score}"
    else:
        # Calculate average of the target_region
        target_mask = scoring["Region"].astype(str).str.contains(target_region, case=False, na=False)
        target_df = scoring[target_mask]
        if target_df.empty:
            return f"ERROR: Target region '{target_region}' not found in data."

        target_vals = pd.to_numeric(target_df[indicator_name], errors="coerce").dropna()
        if target_vals.empty:
            return f"ERROR: No data for '{indicator_name}' in target region '{target_region}'."

        final_target = float(target_vals.mean())
        target_desc = f"average of '{target_region}' ({final_target:.2f})"

    # 3. Calculate Delta
    delta = final_target - current_mean

    # 4. Regression for Total Benchmark Impact
    predicted_delta_total = None
    coef_str = ""
    if "Total_Benchmark" in scoring.columns:
        df_reg = scoring[[indicator_name, "Total_Benchmark"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(df_reg) >= 5:
            X = df_reg[[indicator_name]].values
            y = df_reg["Total_Benchmark"].values
            reg = LinearRegression()
            reg.fit(X, y)
            coef = float(reg.coef_[0])
            predicted_delta_total = coef * delta
            coef_str = (
                f"Regression coef: {coef:.3f} (impact on Total Benchmark per 1 point in {indicator_name})."
            )

    lines = [
        f"Modeling improvement for Region: '{region_name}' on '{indicator_name}'.",
        f"- Current Average: {current_mean:.2f}",
        f"- Target: {target_desc}",
        f"- Required Improvement (Delta): {delta:+.2f} points"
    ]

    if predicted_delta_total is not None:
        lines.append(
            f"Estimated impact on Total Benchmark: {predicted_delta_total:+.2f} points."
        )
        lines.append(f"({coef_str})")
    else:
        lines.append("Could not calculate regression impact due to insufficient data.")

    return "\n".join(lines)


@tool
def regression_indicator_impact(
        indicator_name: str,
        company_keyword: str = "",
        drop_zero_total: bool = True,
        drop_zero_indicator: bool = False,
        fit_intercept: bool = True,
        delta_points: float = 10.0,
        allow_multiple_company_matches: bool = True,
) -> str:
    """
    PREDICTION TOOL (Regression Analysis):
    Fits a simple linear regression: Total_Benchmark ~ <indicator_name> on the dataset
    and (optionally) shows company-specific illustration(s).

    Why this tool is useful:
    - Coefficients can change depending on how you clean the data (e.g., dropping 0-score rows).
      This tool makes those choices explicit via parameters.

    Parameters:
    - indicator_name: Exact column name (e.g., 'Traceability_Risk', 'Purchasing_Practices', 'Remedy').
    - company_keyword: Optional company name/substring to show current values and an illustrative delta.
    - drop_zero_total: If True, exclude rows where Total_Benchmark <= 0.
    - drop_zero_indicator: If True, exclude rows where indicator <= 0.
    - fit_intercept: If True, fit an intercept (default True).
    - delta_points: The hypothetical improvement amount for the indicator (default 10.0).
    - allow_multiple_company_matches: If True, show all matches; if False, error on ambiguity.

    Returns:
    - A readable summary including n, R², intercept, coefficient, and optional company illustration(s).
    """
    # 1) Load data
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    # 2) Validate columns
    if indicator_name not in scoring.columns:
        return f"ERROR: Indicator column '{indicator_name}' not found."
    if "Total_Benchmark" not in scoring.columns:
        return "ERROR: Total_Benchmark column not found."
    if "Company" not in scoring.columns:
        return "ERROR: Company column not found."

    # 3) Build regression dataset with controlled cleaning
    df_reg = scoring[[indicator_name, "Total_Benchmark"]].copy()
    df_reg[indicator_name] = pd.to_numeric(df_reg[indicator_name], errors="coerce")
    df_reg["Total_Benchmark"] = pd.to_numeric(df_reg["Total_Benchmark"], errors="coerce")
    df_reg = df_reg.dropna(subset=[indicator_name, "Total_Benchmark"])

    if drop_zero_total:
        df_reg = df_reg[df_reg["Total_Benchmark"] > 0]
    if drop_zero_indicator:
        df_reg = df_reg[df_reg[indicator_name] > 0]

    n = len(df_reg)
    if n < 5:
        return (
            f"ERROR: Not enough usable rows for regression (n={n}). "
            f"Try changing drop_zero_total/drop_zero_indicator."
        )

    # 4) Fit regression
    X = df_reg[[indicator_name]].values
    y = df_reg["Total_Benchmark"].values

    reg = LinearRegression(fit_intercept=fit_intercept)
    reg.fit(X, y)

    coef = float(reg.coef_[0])
    intercept = float(reg.intercept_) if fit_intercept else 0.0
    r2 = float(reg.score(X, y))

    # 5) Build output
    lines: List[str] = []
    lines.append(f"Regression Analysis: Total_Benchmark ~ {indicator_name}")
    lines.append(
        f"- Settings: drop_zero_total={drop_zero_total}, "
        f"drop_zero_indicator={drop_zero_indicator}, fit_intercept={fit_intercept}"
    )
    lines.append(f"- Stats: n={n}, R²={r2:.3f}, intercept={intercept:.3f}")
    lines.append(f"- Coefficient (slope): {coef:.3f}")
    lines.append(
        f"- Interpretation: +1 point in '{indicator_name}' correlates with "
        f"+{coef:.3f} points in Total_Benchmark (correlational, not causal)."
    )

    # 6) Optional company-specific illustration
    company_keyword = (company_keyword or "").strip()
    if company_keyword:
        comp_df, note = _pick_company_rows(
            scoring,
            company_keyword,
            allow_multiple=allow_multiple_company_matches,
        )

        if note.startswith("ERROR:"):
            lines.append("")
            lines.append(note)
            return "\n".join(lines)

        lines.append("")
        if note:
            lines.append(note)

        if comp_df.empty:
            lines.append(f"No company found matching '{company_keyword}'.")
            return "\n".join(lines)

        # Ensure numeric conversion for display/illustration
        comp_df = comp_df[["Company", indicator_name, "Total_Benchmark"]].copy()
        comp_df[indicator_name] = pd.to_numeric(comp_df[indicator_name], errors="coerce")
        comp_df["Total_Benchmark"] = pd.to_numeric(comp_df["Total_Benchmark"], errors="coerce")

        lines.append(f"Company match(es) for '{company_keyword}': {len(comp_df)} row(s)")

        for _, r in comp_df.iterrows():
            cname = str(r["Company"])
            cur_ind = r[indicator_name]
            cur_tot = r["Total_Benchmark"]

            lines.append(f"- {cname}: {indicator_name}={cur_ind}, Total_Benchmark={cur_tot}")

            if pd.notna(cur_ind) and pd.notna(cur_tot):
                est_change = coef * float(delta_points)
                new_total = float(cur_tot) + est_change
                lines.append(
                    f"  -> If '{indicator_name}' improves by {delta_points:g} points, "
                    f"estimated Total_Benchmark change ≈ {est_change:+.2f} "
                    f"(new total ≈ {new_total:.2f})."
                )
            else:
                lines.append("  -> Cannot illustrate change (missing numeric values).")

    return "\n".join(lines)
