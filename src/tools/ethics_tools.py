from langchain_core.tools import tool

from src.data_loader import get_global_context


@tool
def remedy_region_means() -> str:
    """
    ETHICS AGENT:
    Compute the average Remedy score by Region and present it for ethical comparison.
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    if "Region" not in scoring.columns or "Remedy" not in scoring.columns:
        return "ERROR: Region or Remedy columns missing; cannot analyze regional Remedy differences."

    grp = scoring.groupby("Region")["Remedy"].mean(numeric_only=True)
    if grp.empty:
        return "ERROR: No Remedy data grouped by Region could be computed."

    lines = ["Average Remedy scores by Region:"]
    for region, val in grp.items():
        lines.append(f"- {region}: {float(val):.2f}")
    lines.append(
        "\nThese are Remedy-only scores (not total benchmark), so ethical comparisons focus "
        "specifically on access to remedy and grievance mechanisms."
    )
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


@tool
def high_level_ethics_note(topic: str) -> str:
    """
    ETHICS AGENT:
    Provide a high-level ethical reflection for a given topic or question,
    assuming numeric analysis may be handled by other tools.
    """
    return (
        f"From an ethical perspective on '{topic}', it is important to consider how low scores "
        "in areas like Remedy, Monitoring, or Purchasing Practices might translate into real-world "
        "risks for workers, such as lack of effective grievance mechanisms, weak oversight, or "
        "incentives that push costs and risks down the supply chain. Quantitative patterns in the "
        "KTC dataset should be interpreted alongside qualitative information, stakeholder input, "
        "and the lived experiences of workers."
    )
