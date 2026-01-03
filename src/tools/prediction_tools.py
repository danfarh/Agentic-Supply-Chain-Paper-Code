import pandas as pd
from typing import Optional
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
    - current_year: base year for current data (default 2025 for KTC 2025 dataset)
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
    
    n_years = target_year - current_year
    if n_years < 0:
        return f"ERROR: Target year {target_year} is before current year {current_year}."

    future = cur_mean * ((1 + annual_growth) ** n_years)

    return (
        f"The current average Total Benchmark Score (based on {current_year} data) is {cur_mean:.2f}. "
        f"If it grows by {annual_growth * 100:.1f}% per year for {n_years} years, "
        f"the projected average for {target_year} is about {future:.2f}."
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
        # Run regression on the WHOLE dataset, not just the region
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
