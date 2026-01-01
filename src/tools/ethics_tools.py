from langchain_core.tools import tool

from src.data_loader import get_global_context


def _find_uk_msa_column(df):
    for c in df.columns:
        if "msa" in str(c).lower() or "modern slavery" in str(c).lower(): return c
    return None


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
def uk_msa_region_bias() -> str:
    """
    ETHICS AGENT:
    Check for potential regional bias in UK MSA (Modern Slavery Act) compliance
    using the Non-Scored Research sheet, if a suitable MSA column is present.
    """
    ctx = get_global_context()
    non_scored = ctx.non_scored
    if non_scored is None:
        return "ERROR: Non-Scored Research sheet is not loaded."

    msa_col = _find_uk_msa_column(non_scored)
    if msa_col is None:
        return "No column related to UK MSA / Modern Slavery Act found in Non-Scored Research."

    region_col = None
    for c in non_scored.columns:
        if str(c).strip().lower() == "region":
            region_col = c
            break

    df = non_scored.copy()
    df["msa_yes"] = df[msa_col].astype(str).str.lower().isin(["yes", "y", "true", "1"])

    if region_col:
        grp = df.groupby(region_col)["msa_yes"].mean(numeric_only=True)
        if grp.empty:
            return "No regional distribution could be computed for UK MSA compliance."
        lines = ["Share of 'Yes' responses to UK MSA compliance by Region:"]
        for region, r in grp.items():
            lines.append(f"- {region}: {r * 100:.1f}% of companies marked 'Yes'")
        return "\n".join(lines)
    else:
        yes_rate = float(df["msa_yes"].mean())
        return (
            f"Overall, about {yes_rate * 100:.1f}% of companies indicate 'Yes' for UK MSA compliance. "
            "No Region column was found, so regional bias cannot be evaluated from this sheet."
        )


@tool
def uk_msa_distribution() -> str:
    """
    ETHICS AGENT:
    Show the distribution of raw values in the UK MSA / Modern Slavery Act-related column
    (e.g., Yes / No / Unknown / N/A).
    """
    ctx = get_global_context()
    non_scored = ctx.non_scored
    if non_scored is None:
        return "ERROR: Non-Scored Research sheet is not loaded."

    msa_col = _find_uk_msa_column(non_scored)
    if msa_col is None:
        return "No UK MSA / Modern Slavery Act column found; cannot identify compliance patterns."

    counts = non_scored[msa_col].astype(str).str.strip().value_counts().to_dict()
    lines = ["Distribution of UK MSA compliance values (Non-Scored Research):"]
    for val, cnt in counts.items():
        lines.append(f"- '{val}': {cnt} companies")
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
