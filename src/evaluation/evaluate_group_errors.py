"""Audit saved out-of-fold predictions and decompose fitted OLS score differences."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from src.data_loader import load_data_context
from compare_theme_models import THEMES

OUT = HERE / "outputs" / "group_errors"


def main():
    ctx = load_data_context()
    firms = ctx.scoring.merge(ctx.non_scored[["Company", "Subindustry"]], on="Company",
                              how="left", validate="one_to_one").set_index("Company_ID")
    if len(firms) != 45 or firms[THEMES + ["Total_Benchmark", "Region", "Subindustry"]].isna().any().any():
        raise ValueError("Check benchmark edition, theme columns and company metadata.")
    firms["Sector group"] = np.where(firms.Subindustry.eq("Semiconductors"), "Semiconductors", "Other ICT")
    path = HERE / "outputs" / "theme_model_comparison" / "held_out_predictions.csv"
    if not path.exists():
        raise FileNotFoundError("Run compare_theme_models.py first.")
    p = pd.read_csv(path)
    if (set(p.model) != {"OLS", "Tree"} or set(p.company_id) != set(firms.index)
            or p.duplicated(["model", "repeat", "company_id"]).any()
            or not p.groupby(["model","company_id"]).size().eq(10).all()
            or not p.groupby(["model","repeat"]).size().eq(45).all()
            or not np.isfinite(p.predicted).all()
            or not np.allclose(p.actual, p.company_id.map(firms.Total_Benchmark))):
        raise ValueError("Prediction records do not match the expected repeated evaluation.")
    p["signed"] = p.predicted - p.actual
    p["absolute"] = p.signed.abs()
    by_firm = p.groupby(["model","company_id"])[["absolute","signed"]].mean().reset_index()
    by_firm = by_firm.join(firms[["Region","Sector group"]], on="company_id")
    OUT.mkdir(exist_ok=True, parents=True)
    groups = []
    gaps = []
    parts = []
    x = firms[THEMES]
    reg = LinearRegression().fit(x, firms.Total_Benchmark)
    phi = (x - x.mean()) * reg.coef_
    fitted = pd.Series(reg.predict(x), index=x.index)
    np.testing.assert_allclose(phi.sum(axis=1) + fitted.mean(), fitted, atol=1e-9)
    for field, high, low in [("Region","North America","Asia"),
                              ("Sector group","Semiconductors","Other ICT")]:
        table = by_firm.groupby(["model",field]).agg(n=("company_id","size"),
                            mae=("absolute","mean"), signed_error=("signed","mean")).reset_index()
        table = table.rename(columns={field:"group"})
        table["attribute"] = field
        groups.append(table)
        for model, block in table.groupby("model"):
            gaps.append(dict(attribute=field, model=model, mae_gap=block.mae.max()-block.mae.min(),
                             signed_error_gap=block.signed_error.max()-block.signed_error.min()))
        labels = firms[field]
        diff = phi[labels == high].mean() - phi[labels == low].mean()
        total = fitted[labels == high].mean() - fitted[labels == low].mean()
        np.testing.assert_allclose(diff.sum(), total, atol=1e-9)
        for theme, value in diff.items():
            parts.append(dict(contrast=f"{high} minus {low}", theme=theme,
                              contribution=value, fitted_score_gap=total))
    for filename, table in [("group_metrics",pd.concat(groups)), ("error_gaps",pd.DataFrame(gaps)),
                            ("shapley_decomposition",pd.DataFrame(parts))]:
        table.to_csv(OUT / (filename + ".csv"), index=False)
        print(table.round(4).to_string(index=False))
    print("Saved to", OUT)


if __name__ == "__main__":
    main()
