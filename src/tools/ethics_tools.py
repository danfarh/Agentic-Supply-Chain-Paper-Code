import pandas as pd
from langchain_core.tools import tool

from src.data_loader import get_global_context


@tool
def compute_grouped_average(target_column: str, group_by_column: str) -> str:
    """
    ANALYSIS / ETHICS AGENT:
    Compute the average (mean) of any numeric column grouped by a categorical column.
    Useful for finding regional or industry disparities (e.g., average 'Remedy' by 'Region', or 'Total_Benchmark' by 'Country').
    
    Parameters:
    - target_column: The numeric column to average (e.g., 'Remedy', 'Purchasing Practices').
    - group_by_column: The categorical column to group by (e.g., 'Region', 'Country', 'Subindustry').
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    # Find the exact column names dynamically to avoid case sensitivity issues
    target_match = next((c for c in scoring.columns if target_column.lower() in str(c).lower()), None)
    group_match = next((c for c in scoring.columns if group_by_column.lower() in str(c).lower()), None)

    if not target_match or not group_match:
        return f"ERROR: Could not find columns matching '{target_column}' or '{group_by_column}'."

    df = scoring[[group_match, target_match]].copy()
    df[target_match] = pd.to_numeric(df[target_match], errors='coerce')

    grp = df.groupby(group_match)[target_match].mean().dropna().sort_values(ascending=False)

    if grp.empty:
        return f"ERROR: No valid numeric data found to compute average of '{target_match}' grouped by '{group_match}'."

    lines = [f"Average '{target_match}' grouped by '{group_match}':"]
    for cat, val in grp.items():
        lines.append(f"- {cat}: {val:.2f}")

    return "\n".join(lines)


@tool
def get_categorical_distribution(target_column: str, group_by_column: str = None) -> str:
    """
    ETHICS/DATA AGENT:
    Get the value distribution of a specific categorical column (e.g., 'UK MSA', 'Remedy').
    Optionally group by another column (e.g., 'Region') to check for bias or regional differences.
    """
    ctx = get_global_context()
    df = None

    if ctx.scoring is not None and target_column in ctx.scoring.columns:
        df = ctx.scoring.copy()
    elif ctx.non_scored is not None and target_column in ctx.non_scored.columns:
        df = ctx.non_scored.copy()
    else:
        if ctx.non_scored is not None:
            matched_cols = [c for c in ctx.non_scored.columns if target_column.lower() in str(c).lower()]
            if matched_cols:
                df = ctx.non_scored.copy()
                target_column = matched_cols[0]

    if df is None:
        return f"ERROR: Column containing '{target_column}' not found in loaded data."

    lines = [f"Distribution analysis for '{target_column}':"]

    if group_by_column:
        if group_by_column not in df.columns:
            return f"ERROR: Grouping column '{group_by_column}' not found."

        # Calculate percentage distribution within each group
        grouped = df.groupby(group_by_column)[target_column].value_counts(normalize=True).unstack(fill_value=0) * 100
        lines.append(f"\nGrouped by '{group_by_column}' (percentages %):")
        for index, row in grouped.iterrows():
            stats = ", ".join([f"{col}: {val:.1f}%" for col, val in row.items() if val > 0])
            lines.append(f"- {index}: {stats}")
    else:
        # Overall distribution
        counts = df[target_column].astype(str).str.strip().value_counts()
        percentages = df[target_column].astype(str).str.strip().value_counts(normalize=True) * 100
        for val, count in counts.items():
            pct = percentages[val]
            lines.append(f"- '{val}': {count} companies ({pct:.1f}%)")

    return "\n".join(lines)
