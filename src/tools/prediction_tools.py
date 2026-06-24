from __future__ import annotations

from typing import Optional, List, Tuple, Dict

import pandas as pd
from langchain_core.tools import tool
from sklearn.linear_model import LinearRegression

from src.data_loader import get_global_context


ARTICLE_TABLE8_COEFFICIENTS: Dict[str, float] = {
    # Raw coefficients from the paper/reference answer key.
    # Keep these aligned with your article tables / appendix.
    "Remedy": 0.1019,
    "Purchasing_Practices": 0.0792,
    "Traceability_Risk": 0.2412,
}


def _normalize_text(x: str) -> str:
    return str(x or "").strip().lower()


def _pick_company_rows(
        scoring: pd.DataFrame,
        company_keyword: str,
        *,
        allow_multiple: bool = True,
) -> Tuple[pd.DataFrame, str]:
    if "Company" not in scoring.columns:
        return pd.DataFrame(), "ERROR: Company column not found."

    kw = _normalize_text(company_keyword)
    if not kw:
        return pd.DataFrame(), ""

    companies = scoring["Company"].astype(str)

    exact_mask = companies.map(_normalize_text) == kw
    exact_df = scoring.loc[exact_mask].copy()
    if not exact_df.empty:
        return exact_df, "Matched by exact company name (case-insensitive)."

    contains_mask = companies.str.contains(company_keyword, case=False, na=False, regex=False)
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


def _resolve_column(scoring: pd.DataFrame, column_name: str) -> Optional[str]:
    if column_name in scoring.columns:
        return column_name
    wanted = _normalize_text(column_name).replace(" ", "_")
    for col in scoring.columns:
        norm = _normalize_text(str(col)).replace(" ", "_")
        if norm == wanted or wanted in norm or norm in wanted:
            return col
    return None


def _get_article_coefficient(indicator_name: str) -> Optional[float]:
    key = str(indicator_name).strip()
    if key in ARTICLE_TABLE8_COEFFICIENTS:
        return ARTICLE_TABLE8_COEFFICIENTS[key]
    norm = key.lower().replace(" ", "_")
    for k, v in ARTICLE_TABLE8_COEFFICIENTS.items():
        if k.lower() == norm or k.lower().replace("_", " ") == key.lower():
            return v
    return None


@tool
def project_metric_growth(
        column_name: str,
        annual_growth_rate: float,
        current_year: int = 2025,
        target_year: int = 2027,
) -> str:
    """
    PREDICTION AGENT:
    Project the future industry average for ANY numeric metric using compound annual growth.

    For the benchmark paper's 2027 scenario, use current_year=2025 and target_year=2027.
    Formula: projected_avg = current_avg * (1 + annual_growth_rate) ** (target_year - current_year)
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet not loaded."

    col_match = _resolve_column(scoring, column_name)
    if not col_match:
        return f"ERROR: Column '{column_name}' not found."

    vals = pd.to_numeric(scoring[col_match], errors="coerce").dropna()
    if vals.empty:
        return f"ERROR: No valid numeric data for '{col_match}'."

    current_avg = float(vals.mean())
    years_diff = int(target_year) - int(current_year)

    if years_diff < 0:
        return "ERROR: target_year must be >= current_year."

    projected_avg = current_avg * ((1 + float(annual_growth_rate)) ** years_diff)

    return (
        f"Projection for '{col_match}':\n"
        f"- Baseline year: {current_year}\n"
        f"- Target year: {target_year}\n"
        f"- Number of compounding periods: {years_diff}\n"
        f"- Current Average ({current_year}): {current_avg:.2f}\n"
        f"- Formula: {current_avg:.2f} × (1 + {annual_growth_rate:.4f})^{years_diff}\n"
        f"- Projected Average ({target_year}): {projected_avg:.2f}"
    )


@tool
def model_improvement_impact(
        region_name: str,
        indicator_name: str,
        target_score: Optional[float] = None,
        target_region: Optional[str] = None,
        coefficient_source: str = "article_table8",
) -> str:
    """
    PREDICTION AGENT:
    Calculate how much a Region would need to improve an indicator to reach a target,
    then estimate Total Benchmark impact.

    Use target_region='North America' for the Asia vs North America Purchasing Practices scenario.
    By default, uses Article Table 8 coefficients instead of fitting a fresh regression.
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    if "Region" not in scoring.columns:
        return "ERROR: Region column missing."

    indicator_col = _resolve_column(scoring, indicator_name)
    if not indicator_col:
        return f"ERROR: Indicator column '{indicator_name}' not found."

    if target_score is None and target_region is None:
        return "ERROR: You must provide either 'target_score' or 'target_region'."

    region_mask = scoring["Region"].astype(str).str.contains(region_name, case=False, na=False, regex=False)
    region_df = scoring[region_mask]
    if region_df.empty:
        return f"No companies found in region containing '{region_name}'."

    indicator_vals = pd.to_numeric(region_df[indicator_col], errors="coerce").dropna()
    if indicator_vals.empty:
        return f"No numeric data for indicator '{indicator_col}' in region '{region_name}'."

    current_mean = float(indicator_vals.mean())

    if target_score is not None:
        final_target = float(target_score)
        target_desc = f"fixed target of {final_target:.2f}"
    else:
        target_mask = scoring["Region"].astype(str).str.contains(str(target_region), case=False, na=False, regex=False)
        target_df = scoring[target_mask]
        if target_df.empty:
            return f"ERROR: Target region '{target_region}' not found in data."
        target_vals = pd.to_numeric(target_df[indicator_col], errors="coerce").dropna()
        if target_vals.empty:
            return f"ERROR: No data for '{indicator_col}' in target region '{target_region}'."
        final_target = float(target_vals.mean())
        target_desc = f"average of '{target_region}' ({final_target:.2f})"

    delta = final_target - current_mean

    coef = None
    coef_note = ""
    if coefficient_source == "article_table8":
        coef = _get_article_coefficient(indicator_col)
        if coef is not None:
            coef_note = f"Article Table 8 coefficient for {indicator_col}: {coef:.4f}."
        else:
            coef_note = f"No Article Table 8 coefficient configured for {indicator_col}."
    elif coefficient_source == "fitted_regression":
        if "Total_Benchmark" in scoring.columns:
            df_reg = scoring[[indicator_col, "Total_Benchmark"]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(df_reg) >= 5:
                reg = LinearRegression()
                reg.fit(df_reg[[indicator_col]].values, df_reg["Total_Benchmark"].values)
                coef = float(reg.coef_[0])
                coef_note = f"Fresh fitted regression coefficient: {coef:.4f}."

    predicted_delta_total = coef * delta if coef is not None else None

    lines = [
        f"Modeling improvement for Region: '{region_name}' on '{indicator_col}'.",
        f"- Current {region_name} average: {current_mean:.2f}",
        f"- Target: {target_desc}",
        f"- Required improvement (Delta): {delta:+.2f} points",
        f"- Coefficient source: {coefficient_source}",
        f"- {coef_note}",
    ]

    if predicted_delta_total is not None:
        lines.append(
            f"- Estimated Total Benchmark impact: {delta:.2f} × {coef:.4f} = {predicted_delta_total:+.3f} points."
        )
        lines.append("- Interpretation: this is a small scenario-based estimate, not a causal prediction.")
    else:
        lines.append("- Total Benchmark impact cannot be quantified because no valid coefficient is available.")

    return "\n".join(lines)


@tool
def regression_indicator_impact(
        indicator_name: str,
        company_keyword: str = "",
        drop_zero_total: bool = True,
        drop_zero_indicator: bool = False,
        fit_intercept: bool = True,
        delta_points: Optional[float] = None,
        allow_multiple_company_matches: bool = True,
        coefficient_source: str = "article_table8",
) -> str:
    """
    PREDICTION TOOL:
    Estimate the Total Benchmark score effect of changing an indicator.

    Important safeguards:
    - If delta_points is not explicitly provided, the tool will NOT quantify a score impact.
    - By default it uses Article Table 8 coefficients where configured.
    - A rank cannot be predicted unless a future score distribution/ranking assumption is supplied.
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    indicator_col = _resolve_column(scoring, indicator_name)
    if not indicator_col:
        return f"ERROR: Indicator column '{indicator_name}' not found."
    if "Total_Benchmark" not in scoring.columns:
        return "ERROR: Total_Benchmark column not found."
    if "Company" not in scoring.columns:
        return "ERROR: Company column not found."

    coef = None
    intercept = None
    r2 = None
    n = None
    coefficient_note = ""

    if coefficient_source == "article_table8":
        coef = _get_article_coefficient(indicator_col)
        if coef is None:
            coefficient_note = f"No Article Table 8 coefficient configured for '{indicator_col}'."
        else:
            coefficient_note = f"Using Article Table 8 coefficient for '{indicator_col}': {coef:.4f}."
    elif coefficient_source == "fitted_regression":
        df_reg = scoring[[indicator_col, "Total_Benchmark"]].copy()
        df_reg[indicator_col] = pd.to_numeric(df_reg[indicator_col], errors="coerce")
        df_reg["Total_Benchmark"] = pd.to_numeric(df_reg["Total_Benchmark"], errors="coerce")
        df_reg = df_reg.dropna(subset=[indicator_col, "Total_Benchmark"])
        if drop_zero_total:
            df_reg = df_reg[df_reg["Total_Benchmark"] > 0]
        if drop_zero_indicator:
            df_reg = df_reg[df_reg[indicator_col] > 0]
        n = len(df_reg)
        if n < 5:
            return (
                f"ERROR: Not enough usable rows for regression (n={n}). "
                f"Try changing drop_zero_total/drop_zero_indicator."
            )
        reg = LinearRegression(fit_intercept=fit_intercept)
        reg.fit(df_reg[[indicator_col]].values, df_reg["Total_Benchmark"].values)
        coef = float(reg.coef_[0])
        intercept = float(reg.intercept_) if fit_intercept else 0.0
        r2 = float(reg.score(df_reg[[indicator_col]].values, df_reg["Total_Benchmark"].values))
        coefficient_note = (
            f"Using fresh fitted regression: n={n}, R²={r2:.3f}, "
            f"intercept={intercept:.3f}, coefficient={coef:.4f}."
        )
    else:
        return "ERROR: coefficient_source must be 'article_table8' or 'fitted_regression'."

    lines: List[str] = []
    lines.append(f"Indicator-impact scenario: Total_Benchmark ~ {indicator_col}")
    lines.append(f"- Coefficient source: {coefficient_source}")
    lines.append(f"- {coefficient_note}")
    lines.append("- Interpretation: this is correlational/scenario-based, not causal.")

    company_keyword = (company_keyword or "").strip()
    if company_keyword:
        comp_df, note = _pick_company_rows(
            scoring,
            company_keyword,
            allow_multiple=allow_multiple_company_matches,
        )

        lines.append("")
        if note:
            lines.append(note)

        if comp_df.empty:
            lines.append(f"No company found matching '{company_keyword}'.")
            return "\n".join(lines)

        comp_df = comp_df[["Company", indicator_col, "Total_Benchmark"]].copy()
        comp_df[indicator_col] = pd.to_numeric(comp_df[indicator_col], errors="coerce")
        comp_df["Total_Benchmark"] = pd.to_numeric(comp_df["Total_Benchmark"], errors="coerce")

        lines.append(f"Company match(es) for '{company_keyword}': {len(comp_df)} row(s)")

        for _, r in comp_df.iterrows():
            cname = str(r["Company"])
            cur_ind = r[indicator_col]
            cur_tot = r["Total_Benchmark"]
            lines.append(f"- {cname}: {indicator_col}={cur_ind}, Total_Benchmark={cur_tot}")

            if delta_points is None:
                lines.append(
                    "  -> No numerical score impact is computed because delta_points was not provided. "
                    "Define the assumed change in the indicator before quantifying impact."
                )
            elif coef is not None and pd.notna(cur_tot):
                est_change = float(coef) * float(delta_points)
                new_total = float(cur_tot) + est_change
                lines.append(
                    f"  -> If '{indicator_col}' improves by {float(delta_points):g} points, "
                    f"estimated Total_Benchmark change = {float(coef):.4f} × {float(delta_points):g} "
                    f"= {est_change:+.2f} points (new total ≈ {new_total:.2f})."
                )
                lines.append(
                    "  -> Rank cannot be predicted from this score change alone unless the future score distribution "
                    "and ranking assumptions are provided."
                )

    elif delta_points is not None and coef is not None:
        est_change = float(coef) * float(delta_points)
        lines.append(
            f"If '{indicator_col}' improves by {float(delta_points):g} points, "
            f"estimated Total_Benchmark change = {float(coef):.4f} × {float(delta_points):g} = {est_change:+.2f} points."
        )
    else:
        lines.append("No company or delta_points provided; no numerical scenario impact was computed.")

    return "\n".join(lines)
