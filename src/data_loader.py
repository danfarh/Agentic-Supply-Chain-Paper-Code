import os
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd

from src.config.config import EXCEL_PATH, SCORING_SHEET, DETAILED_SHEET, NON_SCORED_SHEET


@dataclass
class DataContext:
    scoring: Optional[pd.DataFrame] = None
    detailed: Optional[pd.DataFrame] = None
    non_scored: Optional[pd.DataFrame] = None


# Global variable to hold data in memory
_GLOBAL_CTX: Optional[DataContext] = None


def get_global_context() -> DataContext:
    """Lazy-load data context globally to avoid reloading Excel for every tool call."""
    global _GLOBAL_CTX
    if _GLOBAL_CTX is None:
        print("📥 Loading DataContext from Excel...")
        _GLOBAL_CTX = load_data_context()
    return _GLOBAL_CTX


def _normalize_scoring_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the KTC 2025 scoring sheet."""
    df = df_raw.copy()

    if "Company ID" not in df.columns or "Company Name" not in df.columns:
        return df

    company_id_numeric = pd.to_numeric(df["Company ID"], errors="coerce").notna()
    companies = df[company_id_numeric].copy()

    out = pd.DataFrame(index=companies.index)
    out["Company_ID"] = pd.to_numeric(companies["Company ID"], errors="coerce")
    out["Company"] = companies["Company Name"].astype(str).str.strip()

    if "Country" in companies.columns:
        out["Country"] = companies["Country"]
    if "Region" in companies.columns:
        out["Region"] = companies["Region"]

    mc_col = None
    for c in companies.columns:
        if isinstance(c, str) and "market cap" in c.lower():
            mc_col = c
            break
    if mc_col:
        out["Market_Cap"] = pd.to_numeric(companies[mc_col], errors="coerce")

    if df.shape[0] > 1:
        header_row = df.iloc[1]
    else:
        header_row = pd.Series(index=df.columns, dtype=object)

    def map_theme(label: str, canonical: str) -> None:
        nonlocal header_row, companies, out
        mask = header_row.astype(str).str.strip() == label
        if not mask.any():
            return
        col_name = header_row.index[mask.argmax()]
        out[canonical] = pd.to_numeric(companies[col_name], errors="coerce")

    map_theme("Total benchmark score", "Total_Benchmark")
    map_theme("2025 Rank", "Rank_2025")
    map_theme("Commitment & Governance", "Commitment_Governance")
    map_theme("Traceability & Risk Assessment", "Traceability_Risk")
    map_theme("Purchasing Practices", "Purchasing_Practices")
    map_theme("Recruitment", "Recruitment")
    map_theme("Enabling Workers' Rights", "Enabling_Workers")
    map_theme("Monitoring", "Monitoring")
    map_theme("Remedy", "Remedy")

    out.reset_index(drop=True, inplace=True)
    return out


def _normalize_non_scored_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the KTC 2025 Non-scored research sheet (robust to merged headers)."""
    df = df_raw.copy()
    if df.shape[0] < 3:
        return df

    row0 = df.iloc[0].copy().ffill()
    row1 = df.iloc[1].copy()

    new_cols: List[str] = []

    for i in range(len(df.columns)):
        top = str(row0.iloc[i]).strip() if pd.notna(row0.iloc[i]) else ""
        sub = str(row1.iloc[i]).strip() if pd.notna(row1.iloc[i]) else ""
        low = sub.lower()

        name = None

        # First 5 columns: stable company metadata
        if i == 0:
            name = "Company"
        elif i == 1:
            name = "Year_of_inclusion"
        elif i == 2:
            name = "Country"
        elif i == 3:
            name = "Region"
        elif i == 4:
            name = "Market_Cap"

        # UK MSA
        elif "UK Modern Slavery Act" in top:
            if "required to report" in low:
                name = "UK_MSA_required"
            elif "has published a statement" in low:
                name = "UK_MSA_statement"
            elif "comment" in low:
                name = "UK_MSA_comment"
            elif "source" in low:
                name = "UK_MSA_source"

        # CA TSCA
        elif "California Transparency" in top:
            if "required to report" in low:
                name = "CA_TSCA_required"
            elif "has published a statement" in low:
                name = "CA_TSCA_statement"
            elif "comment" in low:
                name = "CA_TSCA_comment"
            elif "source" in low:
                name = "CA_TSCA_source"

        # AU MSA
        elif "Australia Modern Slavery Act" in top:
            if "required to report" in low:
                name = "AU_MSA_required"
            elif "has published a statement" in low:
                name = "AU_MSA_statement"
            elif "comment" in low:
                name = "AU_MSA_comment"
            elif "source" in low:
                name = "AU_MSA_source"

        # High-risk sourcing
        elif "Sourcing from High-Risk Countries" in top:
            if low == "china":
                name = "China"
            elif low == "malaysia":
                name = "Malaysia"
            elif "source" in low:
                name = "HighRisk_Source"

        if not name:
            name = sub if sub else (top if top else f"col_{i}")

        new_cols.append(name)

    df2 = df.copy()
    df2.columns = new_cols
    df2 = df2.iloc[2:].copy()

    # Drop empty company rows
    if "Company" in df2.columns:
        df2 = df2[df2["Company"].notna()].copy()

    # Parse numeric fields
    if "Market_Cap" in df2.columns:
        df2["Market_Cap"] = pd.to_numeric(df2["Market_Cap"], errors="coerce")
    if "Year_of_inclusion" in df2.columns:
        df2["Year_of_inclusion"] = pd.to_numeric(df2["Year_of_inclusion"], errors="coerce")

    bool_cols = {
        "UK_MSA_required", "UK_MSA_statement",
        "CA_TSCA_required", "CA_TSCA_statement",
        "AU_MSA_required", "AU_MSA_statement",
        "China", "Malaysia",
    }
    for col in (bool_cols & set(df2.columns)):
        df2[col] = (
            df2[col].astype(str).str.strip().str.lower()
            .isin(["yes", "yes*", "y", "true", "1"])
        )

    df2.reset_index(drop=True, inplace=True)
    return df2


def load_data_context() -> DataContext:
    ctx = DataContext()
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel file not found at: {EXCEL_PATH}")

    scoring = pd.read_excel(EXCEL_PATH, sheet_name=SCORING_SHEET)
    detailed = pd.read_excel(EXCEL_PATH, sheet_name=DETAILED_SHEET)
    non_scored = pd.read_excel(EXCEL_PATH, sheet_name=NON_SCORED_SHEET)

    ctx.scoring = _normalize_scoring_columns(scoring)
    ctx.detailed = detailed
    ctx.non_scored = _normalize_non_scored_columns(non_scored)

    return ctx
