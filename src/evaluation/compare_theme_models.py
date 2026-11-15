"""Compare theme importance for reconstruction of the 2025 KTC benchmark score."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_data_context

THEMES = ["Commitment_Governance", "Traceability_Risk", "Purchasing_Practices",
          "Recruitment", "Enabling_Workers", "Monitoring", "Remedy"]
OUTPUT = Path(__file__).resolve().parent / "outputs" / "theme_model_comparison"


def main():
    scoring = load_data_context().scoring
    data = scoring.dropna(subset=THEMES + ["Total_Benchmark"]).reset_index(drop=True)
    if data["Company_ID"].duplicated().any():
        raise ValueError("Expected one row per company.")
    print(f"Using {len(data)} of {len(scoring)} companies.")
    x = data[THEMES].to_numpy()
    y = data["Total_Benchmark"].to_numpy()
    models = {"OLS": LinearRegression(),
              "Tree": DecisionTreeRegressor(max_depth=3, min_samples_leaf=3, random_state=42)}
    predictions = []
    importance = []
    complexity = []
    splits = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    for split, (train, test) in enumerate(splits.split(x)):
        repeat, fold = divmod(split, 5)
        for name, model in models.items():
            model.fit(x[train], y[train])
            predicted = model.predict(x[test])
            for i, estimate in zip(test, predicted):
                predictions.append(dict(model=name, repeat=repeat+1, fold=fold+1,
                                        company_id=data.at[i, "Company_ID"], actual=y[i],
                                        predicted=estimate, absolute_error=abs(y[i]-estimate)))
            perm = permutation_importance(model, x[test], y[test],
                                          scoring="neg_mean_absolute_error",
                                          n_repeats=30, random_state=1000+split)
            for theme, change in zip(THEMES, perm.importances_mean):
                importance.append(dict(model=name, repeat=repeat+1, fold=fold+1,
                                       theme=theme, delta_mae=change))
            if name == "Tree":
                complexity.append(dict(repeat=repeat+1, fold=fold+1,
                                       depth=model.get_depth(), leaves=model.get_n_leaves()))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(predictions)
    pred.to_csv(OUTPUT / "held_out_predictions.csv", index=False)
    per_repeat = pred.groupby(["model", "repeat"]).absolute_error.mean()
    errors = per_repeat.groupby("model").agg(["mean", "std"])
    errors.columns = ["cv_mae", "sd_across_repeats"]
    errors.to_csv(OUTPUT / "model_errors.csv")
    folds = pd.DataFrame(importance)
    folds.to_csv(OUTPUT / "fold_importance.csv", index=False)
    ranks = folds.groupby(["model", "repeat", "theme"]).delta_mae.mean().reset_index()
    ranks["rank"] = ranks.groupby(["model", "repeat"]).delta_mae.rank(ascending=False, method="average")
    ranks["top3"] = ranks["rank"] <= 3
    report = ranks.groupby(["model", "theme"]).agg(
        mean_delta_mae=("delta_mae", "mean"), mean_rank=("rank", "mean"),
        top3_fraction=("top3", "mean")).sort_values(["model", "mean_delta_mae"], ascending=[True, False])
    report.to_csv(OUTPUT / "theme_importance.csv")
    trees = pd.DataFrame(complexity)
    trees.to_csv(OUTPUT / "tree_structure.csv", index=False)

    ols = models["OLS"].fit(x, y)
    coefficients = pd.DataFrame({"theme": THEMES, "coefficient": ols.coef_,
                                 "standardised_beta": ols.coef_ * x.std(axis=0) / y.std()})
    coefficients.to_csv(OUTPUT / "full_sample_coefficients.csv", index=False)
    print("Full-sample OLS R-squared:", round(ols.score(x, y), 6))
    print(coefficients.to_string(index=False))
    print(errors.round(4))
    print(report.round(4))
    print("Tree depth and leaves across fits:")
    print(trees[["depth", "leaves"]].agg(["min", "median", "max"]))
    print("Saved results to", OUTPUT)


if __name__ == "__main__":
    main()
