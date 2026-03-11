from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


BASE_DIR = Path(__file__).resolve().parent
START_YEAR = 2010
END_YEAR = 2024

KEEP_COLUMNS = ["Year", "EmployerName", "Position", "RegularPay", "TotalWages"]


def parse_money(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"\s+", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def safe_percent_effect(coef: pd.Series | float) -> pd.Series | float:
    if isinstance(coef, pd.Series):
        out = pd.Series(np.nan, index=coef.index, dtype=float)
        mask = coef.abs() < 20
        out.loc[mask] = (np.exp(coef.loc[mask]) - 1.0) * 100.0
        return out

    if not np.isfinite(coef) or abs(coef) >= 20:
        return math.nan
    return (math.exp(coef) - 1.0) * 100.0


def resolve_year_file(year: int, base_dir: Path) -> Path:
    professors_path = base_dir / "professors" / f"uc_professors_{year}.csv"
    root_path = base_dir / f"uc_professors_{year}.csv"

    if professors_path.exists():
        return professors_path
    if root_path.exists():
        return root_path

    raise FileNotFoundError(f"Could not find file for year {year} in professors/ or repo root.")


def load_panel_data(base_dir: Path, start_year: int, end_year: int) -> tuple[pd.DataFrame, list[Path]]:
    file_paths: list[Path] = []
    frames: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        csv_path = resolve_year_file(year, base_dir)
        file_paths.append(csv_path)

        df = pd.read_csv(csv_path)
        for column in KEEP_COLUMNS:
            if column not in df.columns:
                df[column] = np.nan

        df = df[KEEP_COLUMNS].copy()
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Year"] = df["Year"].fillna(year).astype(int)
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    return panel, file_paths


def map_rank(position: pd.Series) -> pd.Series:
    text = position.fillna("").str.lower()

    rank = np.select(
        [
            text.str.contains("asst") | text.str.contains("assistant"),
            text.str.contains("assoc"),
            text.str.contains("clin"),
            text.str.contains("res") | text.str.contains("research"),
        ],
        ["Assistant", "Associate", "Clinical", "Research"],
        default="Full",
    )

    return pd.Series(rank, index=position.index)


def run_baseline_model(df: pd.DataFrame, outcome: str):
    formula = f"{outcome} ~ post2022 + C(Rank) + C(EmployerName) + C(YearFE)"
    model = smf.ols(formula=formula, data=df).fit(cov_type="HC1")
    return model


def results_to_table(model, model_name: str) -> pd.DataFrame:
    ci = model.conf_int()

    out = pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_err": model.bse.values,
            "t_stat": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_lower": ci[0].values,
            "ci_upper": ci[1].values,
        }
    )

    out["percent_effect_exact"] = safe_percent_effect(out["coefficient"])
    out["percent_effect_approx"] = out["coefficient"] * 100.0
    out["model"] = model_name
    return out


def print_post_summary(model, outcome_label: str) -> None:
    coef = float(model.params.get("post2022", math.nan))
    se = float(model.bse.get("post2022", math.nan))
    pval = float(model.pvalues.get("post2022", math.nan))
    pct = float(safe_percent_effect(coef))

    print(f"\n{outcome_label}: post2022 coefficient summary")
    print(f"  coef = {coef:.4f}, robust_se = {se:.4f}, p_value = {pval:.4g}")
    print(f"  interpretation: post-2022 is associated with about {pct:.2f}% change in {outcome_label}.")


def main() -> None:
    panel, loaded_files = load_panel_data(BASE_DIR, START_YEAR, END_YEAR)

    print("Loaded professor files:")
    for path in loaded_files:
        print(f"  - {path.relative_to(BASE_DIR)}")

    panel["Rank"] = map_rank(panel["Position"])
    panel["post2022"] = (panel["Year"] >= 2022).astype(int)
    panel["YearFE"] = panel["Year"].where(panel["Year"] <= 2021, 2021)

    panel["TotalWages"] = parse_money(panel["TotalWages"])
    panel["RegularPay"] = parse_money(panel["RegularPay"])

    total_df = panel.loc[panel["TotalWages"] > 0].copy()
    total_df["log_total_wages"] = np.log(total_df["TotalWages"])

    regular_df = panel.loc[panel["RegularPay"] > 0].copy()
    regular_df["log_regular_pay"] = np.log(regular_df["RegularPay"])

    model_total = run_baseline_model(total_df, "log_total_wages")
    model_regular = run_baseline_model(regular_df, "log_regular_pay")

    total_table = results_to_table(model_total, "log_total_wages")
    regular_table = results_to_table(model_regular, "log_regular_pay")

    total_out_path = BASE_DIR / "regression_baseline_log_total_wages.csv"
    regular_out_path = BASE_DIR / "regression_baseline_log_regular_pay.csv"

    total_table.to_csv(total_out_path, index=False)
    regular_table.to_csv(regular_out_path, index=False)

    print("\nSaved regression outputs:")
    print(f"  - {total_out_path.name}")
    print(f"  - {regular_out_path.name}")
    print("  - Year fixed effects use 2010-2021 variation so post2022 remains identified.")

    print_post_summary(model_total, "log_total_wages")
    print_post_summary(model_regular, "log_regular_pay")

    print("\nCompact table (key term only):")
    cols = ["term", "coefficient", "std_err", "t_stat", "p_value", "ci_lower", "ci_upper", "percent_effect_exact"]
    print(total_table.loc[total_table["term"] == "post2022", cols].to_string(index=False))
    print(regular_table.loc[regular_table["term"] == "post2022", cols].to_string(index=False))


if __name__ == "__main__":
    main()