from typing import List, Dict, Optional, Any

import pandas as pd
from langchain_core.tools import tool
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

from src.data_loader import get_global_context


def compute_correlation_marketcap_total(scoring: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Compute Pearson correlation between Market_Cap and Total_Benchmark if possible.
    """
    if "Market_Cap" not in scoring.columns or "Total_Benchmark" not in scoring.columns:
        return None

    df = scoring[["Market_Cap", "Total_Benchmark"]].dropna()
    if len(df) < 3:
        return None

    x = df["Market_Cap"].astype(float).values
    y = df["Total_Benchmark"].astype(float).values
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    cov = ((x - x_mean) * (y - y_mean)).sum() / (n - 1)
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    if sx == 0 or sy == 0:
        r = 0.0
    else:
        r = cov / (sx * sy)

    return {
        "n": int(n),
        "r": float(r),
        "p": None
    }


@tool
def get_column_stats(column_name: str) -> str:
    """
    ANALYSIS AGENT (GENERALIZED):
    Compute mean and standard deviation for ANY numeric column in the Scoring sheet.

    Parameters:
    - column_name: the exact normalized column name, e.g.
      "Total_Benchmark", "Remedy", "Monitoring", "Recruitment", etc.
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    if column_name not in scoring.columns:
        return f"ERROR: Column '{column_name}' not found in scoring sheet."

    vals = pd.to_numeric(scoring[column_name], errors="coerce").dropna()
    if vals.empty:
        return f"ERROR: Column '{column_name}' has no numeric data."

    mean_val = float(vals.mean())
    std_val = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    return (
        f"For column '{column_name}', the mean is {mean_val:.2f} "
        f"and the standard deviation is {std_val:.2f}."
    )


@tool
def marketcap_total_correlation() -> str:
    """
    ANALYSIS AGENT:
    Compute the Pearson correlation between Market Cap and Total Benchmark Score.
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    cor = compute_correlation_marketcap_total(scoring)
    if cor is None:
        return "ERROR: Could not compute correlation: not enough data or missing Market_Cap / Total_Benchmark columns."

    r = cor["r"]
    n = cor["n"]
    return (
        f"The Pearson correlation between Market Cap and Total Benchmark Score is r = {r:.3f} "
        f"with sample size n = {n}. "
        "A positive r suggests that higher market cap tends to coincide with higher benchmark scores, "
        "but this does not prove causality."
    )


# @tool
# def marketcap_total_correlation() -> str:
#     """ANALYSIS AGENT: Compute Pearson correlation between Market Cap and Total Benchmark."""
#     ctx = get_global_context()
#     scoring = ctx.scoring
#     if scoring is None: return "ERROR: Data not loaded."
#
#     df = scoring[["Market_Cap", "Total_Benchmark"]].dropna()
#     if len(df) < 3: return "Not enough data."
#
#     r = df["Market_Cap"].corr(df["Total_Benchmark"])
#     return f"Pearson correlation (r) = {r:.3f} (n={len(df)})."


@tool
def perform_clustering(features: List[str], k: int = 3) -> str:
    """
    ANALYSIS AGENT (GENERALIZED):
    Perform K-Means clustering on the specified list of numeric feature columns.

    Parameters:
    - features: list of column names to cluster on, e.g.
        ["Total_Benchmark", "Purchasing_Practices"],
        ["Remedy", "Recruitment"], etc.
    - k: number of clusters (default 3)

    If a 'Company' column exists, example company names are shown for each cluster.
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    if not features:
        return "ERROR: At least one feature must be provided."

    missing = [f for f in features if f not in scoring.columns]
    if missing:
        return f"ERROR: The following feature columns are missing: {missing}"

    df = scoring[features].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < k:
        return "ERROR: Not enough rows with all specified features to run k-means."

    X = df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df_cluster = df.copy()
    df_cluster["Cluster"] = labels

    has_company = "Company" in scoring.columns
    if has_company:
        df_cluster = df_cluster.join(scoring["Company"], how="left")

    lines = [f"k-means clustering (k={k}) on features: {features}"]
    for c_id, grp in df_cluster.groupby("Cluster"):
        if has_company:
            examples = ", ".join(
                grp["Company"].dropna().astype(str).head(3).tolist()
            )
        else:
            examples = "(no company names available)"
        lines.append(
            f"- Cluster {int(c_id)}: count = {len(grp)}, "
            f"example companies: {examples or '(no company names available)'}"
        )
    return "\n".join(lines)


@tool
def theme_medians_by_region(region_substring: str) -> str:
    """
    ANALYSIS AGENT:
    Compute median score for each main theme for companies whose Region contains the given substring.

    Parameters:
    - region_substring: e.g. "North America", "Asia"
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    if "Region" not in scoring.columns:
        return "ERROR: Region column not found in scoring sheet."

    mask = scoring["Region"].astype(str).str.contains(region_substring, case=False, na=False)
    df = scoring[mask]
    if df.empty:
        return f"No companies found for Region containing '{region_substring}'."

    medians: Dict[str, Optional[float]] = {}
    theme_cols = [
        "Total_Benchmark", "Commitment_Governance", "Traceability_Risk",
        "Purchasing_Practices", "Recruitment", "Enabling_Workers",
        "Monitoring", "Remedy"
    ]
    for col in theme_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            medians[col] = float(vals.median()) if len(vals) > 0 else None

    nice_name = {
        "Total_Benchmark": "Total Benchmark",
        "Commitment_Governance": "Commitment & Governance",
        "Traceability_Risk": "Traceability & Risk",
        "Purchasing_Practices": "Purchasing Practices",
        "Recruitment": "Recruitment",
        "Enabling_Workers": "Enabling Workers",
        "Monitoring": "Monitoring",
        "Remedy": "Remedy"
    }

    lines = [f"Median scores for each theme across companies in Region containing '{region_substring}':"]
    for k, v in medians.items():
        if v is not None:
            lines.append(f"- {nice_name.get(k, k)}: {v:.2f}")
    return "\n".join(lines)


@tool
def analyze_sentiment(text: str) -> str:
    """
    ANALYSIS AGENT:
    Analyze the sentiment of a provided text string using TextBlob.
    Returns:
    - Polarity: Float [-1.0, +1.0] (Negative < 0 < Positive)
    - Subjectivity: Float [0.0, 1.0] (0.0 = Objective/Fact, 1.0 = Subjective/Opinion)
    """

    if not text or not text.strip():
        return "ERROR: No text provided for sentiment analysis."

    # Create TextBlob object
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Determine qualitative label
    if polarity > 0.1:
        sentiment_label = "Positive"
    elif polarity < -0.1:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    # Determine subjectivity label
    if subjectivity > 0.5:
        subj_label = "Subjective (Opinion-heavy)"
    else:
        subj_label = "Objective (Factual)"

    return (
        f"Sentiment Analysis Results:\n"
        f"- Sentiment: {sentiment_label} (Polarity Score: {polarity:.3f})\n"
        f"- Nature: {subj_label} (Subjectivity Score: {subjectivity:.3f})\n"
        f"\n(Note: Polarity ranges from -1 to +1. Subjectivity ranges from 0 to 1.)"
    )
