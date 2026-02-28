from typing import List, Dict, Optional

import pandas as pd
from langchain_core.tools import tool
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

from src.data_loader import get_global_context


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


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
def calculate_correlation(column_x: str, column_y: str) -> str:
    """
    ANALYSIS AGENT:
    Compute the Pearson correlation between ANY two numeric columns in the Scoring sheet.

    Parameters:
    - column_x: Exact name of the first column (e.g., 'Market_Cap', 'Total_Benchmark')
    - column_y: Exact name of the second column
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    if column_x not in scoring.columns or column_y not in scoring.columns:
        return f"ERROR: Both '{column_x}' and '{column_y}' must exist in the Scoring sheet."

    # Convert to numeric and drop NA values
    df = scoring[[column_x, column_y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 3:
        return f"ERROR: Not enough valid numeric data to compute correlation between {column_x} and {column_y}."

    r = df[column_x].corr(df[column_y])
    n = len(df)

    return (
        f"Pearson correlation (r) between '{column_x}' and '{column_y}' is {r:.3f} (sample size n={n}).\n"
        "Remember: Correlation does not imply causality."
    )


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
def compare_companies(company_names: List[str]) -> str:
    """
    ANALYSIS AGENT:
    Compare multiple companies side-by-side across the total benchmark score and the 7 main themes.
    This reveals their relative strengths, weaknesses, and identifies the leader in each theme.
    
    Parameters:
    - company_names: A list of company names to compare (e.g., ["Apple", "Samsung", "Sony"]).
    """
    ctx = get_global_context()
    scoring = ctx.scoring
    if scoring is None:
        return "ERROR: Scoring sheet is not loaded."

    if "Company" not in scoring.columns:
        return "ERROR: 'Company' column missing from Scoring sheet."

    # 1. Find and match company names with the database
    matched_companies = {}
    for name in company_names:
        mask = scoring["Company"].astype(str).str.lower().str.contains(_normalize_name(name), na=False)
        df_match = scoring[mask]
        if not df_match.empty:
            row = df_match.iloc[0]
            matched_companies[row["Company"]] = row
        else:
            matched_companies[f"'{name}' (Not Found)"] = None

    valid_companies = {k: v for k, v in matched_companies.items() if v is not None}

    if len(valid_companies) < 2:
        found_list = list(valid_companies.keys())
        return f"ERROR: Need at least two valid companies to compare. Only found: {found_list}"

    # 2. Define core KTC themes
    themes_to_compare = [
        "Total benchmark score",
        "Commitment & Governance",
        "Traceability & Risk Assessment",
        "Purchasing Practices",
        "Recruitment",
        "Enabling Workers' Rights",
        "Monitoring",
        "Remedy"
    ]

    lines = [f"--- Side-by-Side Comparison: {', '.join(valid_companies.keys())} ---"]

    # 3. Compare and find the leader in each theme
    for theme in themes_to_compare:
        matched_col = next((c for c in scoring.columns if theme.lower() in str(c).lower()), None)
        if not matched_col:
            continue

        lines.append(f"\n{matched_col}:")
        theme_scores = {}

        for comp_name, row in valid_companies.items():
            try:
                val = float(row[matched_col])
                theme_scores[comp_name] = val
                lines.append(f"  - {comp_name}: {val:.2f}")
            except (ValueError, TypeError):
                lines.append(f"  - {comp_name}: Data missing/non-numeric")

        if theme_scores:
            max_score = max(theme_scores.values())
            leaders = [c for c, v in theme_scores.items() if v == max_score]
            if len(leaders) == len(theme_scores):
                lines.append(f"  - Insight: All tied at {max_score:.2f}")
            else:
                lines.append(f"  - Insight: Leader(s) -> {', '.join(leaders)} ({max_score:.2f})")

    # 4. Report companies that were not found in the dataset
    missing = [k for k, v in matched_companies.items() if v is None]
    if missing:
        lines.append(f"\nNote: The following companies were not found in the dataset: {', '.join(missing)}")

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
