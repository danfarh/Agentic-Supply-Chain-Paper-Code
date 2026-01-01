import pandas as pd
from langchain_core.tools import tool
from sklearn.linear_model import LinearRegression

from src.data_loader import get_global_context


@tool
def project_total_benchmark(
        target_year: int = 2027,
        current_year: int = 2025,
        annual_growth: float = 0.05
) -> str:
    """
    PREDICTION AGENT:
    Project the industry average Total_Benchmark Score from current_year to target_year
    assuming a constant annual growth rate.

    Parameters:
    - target_year: e.g. 2027
    - current_year: base year for current data (default 2025)
    - annual_growth: e.g. 0.05 for 5% annual growth
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    if "Total_Benchmark" not in scoring.columns:
        return "ERROR: Total_Benchmark column not found."

    vals = pd.to_numeric(scoring["Total_Benchmark"], errors="coerce").dropna()
    if vals.empty:
        return "ERROR: No Total_Benchmark values available."

    cur_mean = float(vals.mean())
    n_years = max(0, target_year - current_year)
    future = cur_mean * ((1 + annual_growth) ** n_years)

    return (
        f"The current average Total Benchmark Score (year {current_year}) is {cur_mean:.2f}. "
        f"If it grows by {annual_growth * 100:.1f}% per year for {n_years} years, "
        f"the projected average for {target_year} is about {future:.2f}."
    )


@tool
def model_improvement_impact(region_name: str, indicator_name: str, target_score: float) -> str:
    """
    PREDICTION AGENT (GENERALIZED):
    Calculate how much a specific Region would need to improve a specific indicator
    to reach a target_score, and optionally estimate the impact on Total_Benchmark via
    a simple linear regression if data is available.

    Parameters:
    - region_name: e.g. "Asia", "Europe", "North America"
    - indicator_name: e.g. "Purchasing_Practices", "Recruitment", "Remedy"
    - target_score: desired average score for that indicator in that region, e.g. 45.0
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    if "Region" not in scoring.columns:
        return "ERROR: Region column missing; cannot model regional improvement."
    if indicator_name not in scoring.columns:
        return f"ERROR: Indicator column '{indicator_name}' not found; cannot model improvement."

    region_mask = scoring["Region"].astype(str).str.contains(region_name, case=False, na=False)
    region_df = scoring[region_mask]
    if region_df.empty:
        return f"No companies found in region containing '{region_name}'."

    indicator_vals = pd.to_numeric(region_df[indicator_name], errors="coerce").dropna()
    if indicator_vals.empty:
        return f"No numeric data for indicator '{indicator_name}' in region '{region_name}'."

    current_mean = float(indicator_vals.mean())
    delta = target_score - current_mean

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
                f"The regression coefficient linking '{indicator_name}' to Total_Benchmark "
                f"is about {coef:.3f} (per one-point change in {indicator_name})."
            )

    lines = [
        f"In region '{region_name}', the current average for indicator '{indicator_name}' is {current_mean:.2f}.",
        f"To reach the target of {target_score:.2f}, the average would need to increase by {delta:.2f} points."
    ]

    if predicted_delta_total is not None:
        lines.append(
            f"Based on a simple linear regression, this could translate into an approximate "
            f"change of {predicted_delta_total:.2f} points in the Total Benchmark on average "
            "(with all caveats about model simplicity)."
        )
        if coef_str:
            lines.append(coef_str)
    else:
        lines.append(
            "A reliable regression linking this indicator to Total_Benchmark could not be established, "
            "so the impact on Total_Benchmark is not quantified."
        )

    return "\n".join(lines)


@tool
def regression_indicator_impact(indicator_name: str, company_keyword: str = "") -> str:
    """
    PREDICTION AGENT (GENERALIZED):
    Fit a simple linear regression Total_Benchmark ~ <indicator_name> on all companies
    and optionally report the current indicator value for a company whose name contains
    company_keyword (if provided).

    Parameters:
    - indicator_name: e.g. "Traceability_Risk", "Purchasing_Practices", "Remedy"
    - company_keyword: e.g. "Apple", "Samsung", "Amazon" (optional)
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    if indicator_name not in scoring.columns:
        return f"ERROR: Indicator column '{indicator_name}' not found."

    if "Total_Benchmark" not in scoring.columns:
        return "ERROR: Total_Benchmark column not found."

    df_reg = scoring[[indicator_name, "Total_Benchmark"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df_reg) < 5:
        return (
            f"ERROR: Not enough data to fit regression between '{indicator_name}' and Total_Benchmark "
            f"(only {len(df_reg)} usable rows)."
        )

    X = df_reg[[indicator_name]].values
    y = df_reg["Total_Benchmark"].values
    reg = LinearRegression()
    reg.fit(X, y)
    coef = float(reg.coef_[0])

    lines = [
        f"The regression coefficient linking '{indicator_name}' to Total_Benchmark is about {coef:.3f} "
        f"(per one-point change in {indicator_name})."
    ]

    if company_keyword and "Company" in scoring.columns:
        comp_mask = scoring["Company"].astype(str).str.contains(company_keyword, case=False, na=False)
        comp_df = scoring[comp_mask]
        if comp_df.empty:
            lines.append(
                f"No company with name containing '{company_keyword}' was found to illustrate this impact."
            )
        else:
            val = pd.to_numeric(comp_df[indicator_name], errors="coerce").iloc[0]
            if pd.isna(val):
                lines.append(
                    f"A company matching '{company_keyword}' was found, but its '{indicator_name}' value is missing."
                )
            else:
                lines.append(
                    f"A company matching '{company_keyword}' has '{indicator_name}' ≈ {float(val):.2f}. "
                    "If this company improves its indicator score by Δ, its Total_Benchmark is expected to change "
                    f"by approximately {coef:.3f} × Δ, according to this simple linear model."
                )

    lines.append(
        "This regression is purely correlational and should not be interpreted as causal without further evidence."
    )

    return "\n".join(lines)
