"""
export_figures.py
-----------------
Saves all publication-quality figures and regression tables from the
UC professor salary analysis to /figures/.

Run from the project root:
    python export_figures.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from scipy import stats as scipy_stats
from pathlib import Path

# ── Theme ──────────────────────────────────────────────────────────────────────
try:
    import morethemes as mt
    mt.set_theme("economist")
    THEME = "economist"
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid")
    THEME = "seaborn"

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
PROF_DIR = BASE_DIR / "professors"
FIG_DIR  = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

DPI = 300

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

KEEP_COLS = ["Year", "EmployerName", "Position", "RegularPay", "TotalWages"]

def load_panel() -> pd.DataFrame:
    """Load, deduplicate, and annotate the 2012-2024 professor panel."""
    frames = []
    for year in range(2012, 2025):
        path = PROF_DIR / f"uc_professors_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        df = pd.read_csv(path)
        for col in KEEP_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[KEEP_COLS].copy()
        df["Year"] = year
        df = df.drop_duplicates()
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)

    # Rank categorization (same logic as main.ipynb)
    pos_lower = panel["Position"].fillna("").str.lower()
    panel["Rank"] = np.select(
        [
            pos_lower.str.contains("assistant") | pos_lower.str.contains("asst"),
            pos_lower.str.contains("associate")  | pos_lower.str.contains("assoc"),
            pos_lower.str.contains("clinical")   | pos_lower.str.contains("clin"),
            pos_lower.str.contains("research")   | pos_lower.str.contains("res"),
        ],
        ["Assistant", "Associate", "Clinical", "Research"],
        default="Full",
    )

    # CSVs in professors/ are already professor-filtered; no regex needed here.
    panel["TotalWages"] = pd.to_numeric(panel["TotalWages"], errors="coerce")
    panel["RegularPay"] = pd.to_numeric(panel["RegularPay"], errors="coerce")
    panel["post2022"]   = (panel["Year"] >= 2022).astype(int)

    return panel


def make_subsets(panel: pd.DataFrame):
    """Return (panel_tw, panel_rp) for the two outcome regressions."""
    panel_tw = panel[panel["TotalWages"] > 0].copy()
    panel_tw["log_total_wages"] = np.log(panel_tw["TotalWages"])

    panel_rp = panel[panel["RegularPay"] > 0].copy()
    panel_rp["log_regular_pay"] = np.log(panel_rp["RegularPay"])

    return panel_tw, panel_rp


print("Loading data …")
panel    = load_panel()
panel_tw, panel_rp = make_subsets(panel)
print(f"  panel_tw rows : {len(panel_tw):,}")
print(f"  panel_rp rows : {len(panel_rp):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  YEARLY SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def yearly_stats(panel_tw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in sorted(panel_tw["Year"].unique()):
        d = panel_tw.loc[panel_tw["Year"] == year, "TotalWages"]
        rows.append({
            "Year":   year,
            "Count":  len(d),
            "Mean":   d.mean(),
            "Median": d.median(),
            "Std":    d.std(),
            "Q25":    d.quantile(0.25),
            "Q75":    d.quantile(0.75),
        })
    df = pd.DataFrame(rows)
    df["YoY_Change_Pct"] = df["Mean"].pct_change() * 100
    return df

yearly_df = yearly_stats(panel_tw)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  VARIATION BY RANK
# ══════════════════════════════════════════════════════════════════════════════

RANKS = ["Assistant", "Associate", "Full", "Clinical", "Research"]

def variation_by_rank(panel_tw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in sorted(panel_tw["Year"].unique()):
        for rank in RANKS:
            d = panel_tw.loc[
                (panel_tw["Year"] == year) & (panel_tw["Rank"] == rank), "TotalWages"
            ]
            if len(d) < 2:
                continue
            mean_sal = d.mean()
            std_sal  = d.std()
            rows.append({
                "Year": year,
                "Rank": rank,
                "Count": len(d),
                "Mean":  mean_sal,
                "Std":   std_sal,
                "CV%":   (std_sal / mean_sal * 100) if mean_sal > 0 else np.nan,
                "Q25":   d.quantile(0.25),
                "Q75":   d.quantile(0.75),
                "IQR":   d.quantile(0.75) - d.quantile(0.25),
            })
    return pd.DataFrame(rows)

variation_df = variation_by_rank(panel_tw)

# ══════════════════════════════════════════════════════════════════════════════
# 4.  REGRESSIONS
# ══════════════════════════════════════════════════════════════════════════════

print("Running regressions …")

# Baseline
model_tw = smf.ols(
    "log_total_wages ~ post2022 + C(Rank) + C(EmployerName) + Year",
    data=panel_tw,
).fit(cov_type="HC1")

model_rp = smf.ols(
    "log_regular_pay ~ post2022 + C(Rank) + C(EmployerName) + Year",
    data=panel_rp,
).fit(cov_type="HC1")

# DiD / compression
model_did = smf.ols(
    "log_total_wages ~ post2022 * C(Rank) + C(EmployerName) + Year",
    data=panel_tw,
).fit(cov_type="HC1")

# Within-rank
rank_models = {}
for rank in RANKS:
    sub = panel_tw[panel_tw["Rank"] == rank].copy()
    if len(sub) < 30:
        continue
    rank_models[rank] = smf.ols(
        "log_total_wages ~ post2022 + C(EmployerName) + Year",
        data=sub,
    ).fit(cov_type="HC1")

print("  Regressions complete.")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  LEVENE TEST
# ══════════════════════════════════════════════════════════════════════════════

levene_rows = []
for rank in RANKS:
    pre  = panel_tw.loc[(panel_tw["Rank"] == rank) & (panel_tw["Year"] < 2022),  "TotalWages"].values
    post = panel_tw.loc[(panel_tw["Rank"] == rank) & (panel_tw["Year"] >= 2022), "TotalWages"].values
    if len(pre) < 2 or len(post) < 2:
        continue
    stat, pval = scipy_stats.levene(pre, post)
    var_pre  = np.var(pre,  ddof=1)
    var_post = np.var(post, ddof=1)
    levene_rows.append({
        "Rank":              rank,
        "Pre-2022 Variance": var_pre,
        "Post-2022 Variance": var_post,
        "Variance Ratio":    var_post / var_pre,
        "Levene Stat":       stat,
        "p-value":           pval,
        "Significant":       "Yes" if pval < 0.05 else "No",
    })
levene_df = pd.DataFrame(levene_rows)

# ══════════════════════════════════════════════════════════════════════════════
# 6.  VIF
# ══════════════════════════════════════════════════════════════════════════════

def compute_vif(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    X = df[cols].copy().dropna()
    for c in cols:
        if X[c].dtype == object:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    X = X.astype(float)
    X_const = sm.add_constant(X)
    return pd.DataFrame({
        "Predictor": X_const.columns,
        "VIF": [variance_inflation_factor(X_const.values, i)
                for i in range(X_const.shape[1])],
    })[lambda d: d["Predictor"] != "const"].reset_index(drop=True)

vif_df = compute_vif(panel_tw, ["Year", "post2022", "Rank", "EmployerName"])

# ══════════════════════════════════════════════════════════════════════════════
# 7.  CV BY RANK & YEAR (pivot table for table_cv_by_rank_year)
# ══════════════════════════════════════════════════════════════════════════════

cv_pivot = variation_df.pivot(index="Year", columns="Rank", values="CV%").round(2)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

RANK_COLORS = {
    "Assistant": "#1f77b4",
    "Associate": "#ff7f0e",
    "Full":      "#2ca02c",
    "Clinical":  "#d62728",
    "Research":  "#9467bd",
}

def savefig(name: str):
    path = FIG_DIR / name
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1: fig_salary_trends.png  (4-panel)
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating fig_salary_trends.png …")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("UC Professor Salary Trends, 2012–2024 (Nominal)", fontsize=15, fontweight="bold", y=1.01)

years_num = yearly_df["Year"].values.astype(int)

# Panel 1 — mean with ±1 std band
ax = axes[0, 0]
ax.plot(years_num, yearly_df["Mean"], marker="o", linewidth=2.5, markersize=7,
        color="#1f6eb5", label="Mean salary")
ax.fill_between(years_num,
                yearly_df["Mean"] - yearly_df["Std"],
                yearly_df["Mean"] + yearly_df["Std"],
                alpha=0.18, color="#1f6eb5", label="±1 SD")
ax.axvline(2022, color="#c0392b", linestyle="--", linewidth=1.8, label="2022 threshold")
ax.set_title("Mean Total Wages with ±1 SD Band", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Total Wages (nominal $)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2 — YoY % change bar chart
ax = axes[0, 1]
yoy = yearly_df["YoY_Change_Pct"].fillna(0).values
colors_bar = ["#27ae60" if v >= 0 else "#c0392b" for v in yoy]
ax.bar(years_num, yoy, color=colors_bar, alpha=0.8, edgecolor="white", linewidth=0.5)
ax.axvline(2022, color="#c0392b", linestyle="--", linewidth=1.8, label="2022 threshold")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Year-over-Year % Change in Mean Salary", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("YoY Change (%)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# Panel 3 — quartile trends
ax = axes[1, 0]
ax.plot(years_num, yearly_df["Q25"],    marker="^", linewidth=2, markersize=6, color="#e67e22", label="Q25")
ax.plot(years_num, yearly_df["Median"], marker="s", linewidth=2.5, markersize=7, color="#27ae60", label="Median")
ax.plot(years_num, yearly_df["Q75"],    marker="^", linewidth=2, markersize=6, color="#8e44ad", label="Q75")
ax.axvline(2022, color="#c0392b", linestyle="--", linewidth=1.8, label="2022 threshold")
ax.set_title("Quartile Salary Trends (Q25 / Median / Q75)", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Total Wages (nominal $)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 4 — professor count over time
ax = axes[1, 1]
ax.plot(years_num, yearly_df["Count"], marker="o", linewidth=2.5, markersize=7,
        color="#34495e", label="Professor count")
ax.axvline(2022, color="#c0392b", linestyle="--", linewidth=1.8, label="2022 threshold")
ax.set_title("Professor Observation Count Over Time", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Observations")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
savefig("fig_salary_trends.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2: fig_cv_by_rank.png  (4-panel)
# ══════════════════════════════════════════════════════════════════════════════

print("Generating fig_cv_by_rank.png …")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Salary Dispersion by Professor Rank, 2012–2024", fontsize=15,
             fontweight="bold", y=1.01)

# CV pre vs post for bar chart
cv_pre_post = {}
for rank in RANKS:
    rd = variation_df[variation_df["Rank"] == rank].sort_values("Year")
    pre_cv  = rd.loc[rd["Year"] < 2022,  "CV%"].mean()
    post_cv = rd.loc[rd["Year"] >= 2022, "CV%"].mean()
    cv_pre_post[rank] = {"pre": pre_cv, "post": post_cv, "change": post_cv - pre_cv}

# Panel 1 — CV% over time
ax = axes[0, 0]
for rank in RANKS:
    rd = variation_df[variation_df["Rank"] == rank].sort_values("Year")
    ax.plot(rd["Year"], rd["CV%"], marker="o", linewidth=2, markersize=5,
            color=RANK_COLORS[rank], label=rank)
ax.axvline(2022, color="#c0392b", linestyle="--", linewidth=1.8, label="2022 threshold")
ax.set_title("Coefficient of Variation (CV%) by Rank", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("CV (%)")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Panel 2 — Std dev over time
ax = axes[0, 1]
for rank in RANKS:
    rd = variation_df[variation_df["Rank"] == rank].sort_values("Year")
    ax.plot(rd["Year"], rd["Std"], marker="s", linewidth=2, markersize=5,
            color=RANK_COLORS[rank], label=rank)
ax.axvline(2022, color="#c0392b", linestyle="--", linewidth=1.8, label="2022 threshold")
ax.set_title("Standard Deviation of Salary by Rank", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Std Dev (nominal $)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Panel 3 — IQR over time
ax = axes[1, 0]
for rank in RANKS:
    rd = variation_df[variation_df["Rank"] == rank].sort_values("Year")
    ax.plot(rd["Year"], rd["IQR"], marker="^", linewidth=2, markersize=5,
            color=RANK_COLORS[rank], label=rank)
ax.axvline(2022, color="#c0392b", linestyle="--", linewidth=1.8, label="2022 threshold")
ax.set_title("Interquartile Range (IQR) by Rank", fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("IQR (nominal $)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Panel 4 — horizontal bar: CV change pre vs post-2022
ax = axes[1, 1]
rank_order = sorted(cv_pre_post.keys(), key=lambda r: cv_pre_post[r]["change"])
changes    = [cv_pre_post[r]["change"] for r in rank_order]
bar_colors = ["#c0392b" if v > 0 else "#27ae60" for v in changes]
bars = ax.barh(rank_order, changes, color=bar_colors, alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("CV% Change: Pre-2022 → Post-2022 Average", fontweight="bold")
ax.set_xlabel("ΔCV% (Post − Pre)")
for bar, val in zip(bars, changes):
    ax.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}pp", va="center", ha="left" if val >= 0 else "right", fontsize=9)
ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
savefig("fig_cv_by_rank.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3: fig_pre_post_distributions.png  (5-panel overlaid histograms)
# ══════════════════════════════════════════════════════════════════════════════

print("Generating fig_pre_post_distributions.png …")

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle("Salary Distribution by Rank: Pre-2022 vs Post-2022 (Total Wages, Nominal)",
             fontsize=13, fontweight="bold", y=1.02)

for idx, rank in enumerate(RANKS):
    ax  = axes[idx]
    pre  = panel_tw.loc[(panel_tw["Rank"] == rank) & (panel_tw["Year"] < 2022),  "TotalWages"].values
    post = panel_tw.loc[(panel_tw["Rank"] == rank) & (panel_tw["Year"] >= 2022), "TotalWages"].values

    if len(pre) == 0 or len(post) == 0:
        ax.set_visible(False)
        continue

    # Shared bin edges
    all_vals  = np.concatenate([pre, post])
    bins      = np.linspace(np.percentile(all_vals, 1), np.percentile(all_vals, 99), 50)

    ax.hist(pre,  bins=bins, alpha=0.55, color="#2980b9", label="Pre-2022",  edgecolor="none")
    ax.hist(post, bins=bins, alpha=0.55, color="#e74c3c", label="Post-2022", edgecolor="none")

    mean_pre  = np.mean(pre)
    mean_post = np.mean(post)
    ax.axvline(mean_pre,  color="#1a5276", linestyle="--", linewidth=1.8,
               label=f"Mean pre: ${mean_pre:,.0f}")
    ax.axvline(mean_post, color="#922b21", linestyle="--", linewidth=1.8,
               label=f"Mean post: ${mean_post:,.0f}")

    ax.set_title(f"{rank}\nProfessors", fontweight="bold", fontsize=10)
    ax.set_xlabel("Total Wages ($)", fontsize=8)
    ax.set_ylabel("Frequency" if idx == 0 else "", fontsize=8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}k"))
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)

plt.tight_layout()
savefig("fig_pre_post_distributions.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4: fig_headcount_by_rank.png
# ══════════════════════════════════════════════════════════════════════════════

print("Generating fig_headcount_by_rank.png …")

by_year_rank = (
    panel_tw.groupby(["Year", "Rank"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

fig, ax = plt.subplots(figsize=(12, 6))
for rank in RANKS:
    if rank in by_year_rank.columns:
        ax.plot(by_year_rank.index, by_year_rank[rank],
                marker="o", linewidth=2.2, markersize=6,
                color=RANK_COLORS[rank], label=rank)

ax.axvline(2022, color="#c0392b", linestyle="--", linewidth=1.8, label="2022 threshold")
ax.set_title("Professor Headcount by Rank, 2012–2024", fontsize=14, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Observations")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(title="Rank", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
savefig("fig_headcount_by_rank.png")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE RENDERING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

SERIF_FONT = "DejaVu Serif"

def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def render_regression_table(
    rows: list[dict],
    col_headers: list[str],
    title: str,
    note: str,
    save_name: str,
    col_widths: list[float] | None = None,
):
    """
    Render a professional regression table as a .png image.

    rows        : list of dicts; key=column header, value=cell string
    col_headers : list of column header strings (determines column order)
    title       : bold title above the table
    note        : note text below the table
    save_name   : filename in FIG_DIR
    col_widths  : relative column widths (defaults to equal)
    """
    n_rows = len(rows)
    n_cols = len(col_headers)

    if col_widths is None:
        col_widths = [1.0] * n_cols

    # Figure sizing — journal style: narrow and tall
    row_height    = 0.38   # inches per data row
    header_height = 0.46   # inches for header row
    title_height  = 0.60   # inches for title
    note_height   = 0.30   # inches per note line

    n_note_lines = len(note.split("\n"))
    total_height = title_height + header_height + n_rows * row_height + n_note_lines * note_height + 0.3
    fig_width    = 7.0     # fixed journal column width

    fig = plt.figure(figsize=(fig_width, total_height))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    # ── Title ────────────────────────────────────────────────────────────────
    y_cursor = total_height - 0.08
    ax.text(fig_width / 2, y_cursor, title,
            ha="center", va="top",
            fontsize=11, fontweight="bold", fontfamily=SERIF_FONT)
    y_cursor -= title_height

    # ── Compute column x positions ────────────────────────────────────────────
    total_w   = sum(col_widths)
    x_margins = 0.3          # left/right margin in inches
    usable_w  = fig_width - 2 * x_margins
    x_starts  = []
    x_centers = []
    cumulative = x_margins
    for w in col_widths:
        scaled = w / total_w * usable_w
        x_starts.append(cumulative)
        x_centers.append(cumulative + scaled / 2)
        cumulative += scaled

    # ── Horizontal rule above header ────────────────────────────────────────
    ax.axhline(y_cursor, xmin=x_margins / fig_width,
               xmax=(fig_width - x_margins) / fig_width,
               color="black", linewidth=1.2)

    # ── Header row ────────────────────────────────────────────────────────────
    y_header = y_cursor - header_height / 2
    for col_i, (header, xc) in enumerate(zip(col_headers, x_centers)):
        ha = "left" if col_i == 0 else "center"
        x  = x_starts[col_i] + 0.05 if col_i == 0 else xc
        ax.text(x, y_header, header,
                ha=ha, va="center",
                fontsize=9, fontweight="bold", fontfamily=SERIF_FONT)

    y_cursor -= header_height

    # ── Horizontal rule below header ─────────────────────────────────────────
    ax.axhline(y_cursor, xmin=x_margins / fig_width,
               xmax=(fig_width - x_margins) / fig_width,
               color="black", linewidth=0.8)

    # ── Data rows ─────────────────────────────────────────────────────────────
    for row_i, row_data in enumerate(rows):
        y_row = y_cursor - row_height * (row_i + 0.5)
        # Alternating row shade
        if row_i % 2 == 1:
            shade_rect = mpatches.FancyBboxPatch(
                (x_margins, y_cursor - row_height * (row_i + 1)),
                fig_width - 2 * x_margins, row_height,
                boxstyle="square,pad=0", linewidth=0,
                facecolor="#f5f5f5", zorder=0,
            )
            ax.add_patch(shade_rect)

        for col_i, (header, xc) in enumerate(zip(col_headers, x_centers)):
            cell_val = str(row_data.get(header, ""))
            ha = "left" if col_i == 0 else "center"
            x  = x_starts[col_i] + 0.05 if col_i == 0 else xc
            ax.text(x, y_row, cell_val,
                    ha=ha, va="center",
                    fontsize=8.5, fontfamily=SERIF_FONT)

    y_cursor -= row_height * n_rows

    # ── Bottom rule ───────────────────────────────────────────────────────────
    ax.axhline(y_cursor, xmin=x_margins / fig_width,
               xmax=(fig_width - x_margins) / fig_width,
               color="black", linewidth=1.2)

    # ── Note ─────────────────────────────────────────────────────────────────
    y_note = y_cursor - 0.08
    ax.text(x_margins, y_note, note,
            ha="left", va="top",
            fontsize=7.5, fontfamily=SERIF_FONT, color="#333333",
            linespacing=1.4)

    fig.patch.set_facecolor("white")
    path = FIG_DIR / save_name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def model_to_rows(model, n_obs: int, include_campus_fe: bool = True) -> tuple[list[dict], str]:
    """
    Convert a statsmodels OLS result into rows for render_regression_table.
    - Rank dummies are shown explicitly with coefficients.
    - Campus dummies and Year are omitted from rows; noted in footnote.
    Returns (rows, note_string, col_headers).
    """
    params  = model.params
    bse     = model.bse
    tvals   = model.tvalues
    pvals   = model.pvalues
    ci      = model.conf_int()
    r2      = model.rsquared
    nobs    = int(model.nobs)

    rows = []

    COL_HEADERS_REG = ["Variable", "Coef.", "Std. Err.", "t-stat", "p-value", "95% CI"]

    for term in params.index:
        term_lower = term.lower()

        # Skip campus dummies entirely — noted in footnote
        if "employername" in term_lower:
            continue

        # Skip Year — noted in footnote
        if term == "Year":
            continue

        # Skip Intercept
        if term == "Intercept":
            continue

        coef   = params[term]
        se     = bse[term]
        t      = tvals[term]
        p      = pvals[term]
        ci_lo  = ci.loc[term, 0]
        ci_hi  = ci.loc[term, 1]
        sig    = stars(p)

        # Clean up term labels
        label = term
        label = label.replace("C(Rank)[T.", "").replace("]", "")
        label = label.replace("post2022:C(Rank)[T.", "post2022 × ").replace("]", "")

        rows.append({
            "Variable":    label + sig,
            "Coef.":       f"{coef:.4f}",
            "Std. Err.":   f"{se:.4f}",
            "t-stat":      f"{t:.3f}",
            "p-value":     f"{p:.4f}",
            "95% CI":      f"[{ci_lo:.4f}, {ci_hi:.4f}]",
        })

    # Indicator rows for suppressed controls
    for label in ("Campus FE", "Year Trend"):
        rows.append({
            "Variable":  label,
            "Coef.":     "Yes",
            "Std. Err.": "",
            "t-stat":    "",
            "p-value":   "",
            "95% CI":    "",
        })

    note = (
        f"N = {nobs:,}   R² = {r2:.4f}\n"
        "Reference category: Assistant Professor.\n"
        "HC1 heteroskedasticity-robust standard errors.   Stars: *** p<0.001  ** p<0.01  * p<0.05"
    )
    return rows, note, COL_HEADERS_REG


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1: table_baseline_total_wages.png
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating regression tables …")

REG_COL_WIDTHS = [2.8, 1.0, 1.0, 1.0, 1.0, 1.4]   # Variable | Coef | SE | t | p | CI

rows, note, hdrs = model_to_rows(model_tw, int(model_tw.nobs))
render_regression_table(
    rows=rows,
    col_headers=hdrs,
    title="Table 1. Baseline OLS: log(Total Wages) ~ post2022 + Rank FE + Campus FE + Year",
    note=note,
    save_name="table_baseline_total_wages.png",
    col_widths=REG_COL_WIDTHS,
)

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2: table_baseline_regular_pay.png
# ══════════════════════════════════════════════════════════════════════════════

rows, note, hdrs = model_to_rows(model_rp, int(model_rp.nobs))
render_regression_table(
    rows=rows,
    col_headers=hdrs,
    title="Table 2. Baseline OLS: log(Regular Pay) ~ post2022 + Rank FE + Campus FE + Year",
    note=note,
    save_name="table_baseline_regular_pay.png",
    col_widths=REG_COL_WIDTHS,
)

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3: table_did_compression.png
# ══════════════════════════════════════════════════════════════════════════════

# For DiD table only show interaction terms + post2022 main + condensed FEs
params_did   = model_did.params
bse_did      = model_did.bse
tvals_did    = model_did.tvalues
pvals_did    = model_did.pvalues
ci_did       = model_did.conf_int()

did_rows = []
priority_terms = ["Intercept", "post2022", "Year"]
interaction_terms_did = [t for t in params_did.index
                         if "post2022:c(rank)" in t.lower()]
rank_main_terms = [t for t in params_did.index
                   if "c(rank)" in t.lower() and "post2022" not in t.lower()]

def did_row(term):
    coef  = params_did[term]
    se    = bse_did[term]
    t     = tvals_did[term]
    p     = pvals_did[term]
    ci_lo = ci_did.loc[term, 0]
    ci_hi = ci_did.loc[term, 1]
    sig   = stars(p)
    label = term
    label = label.replace("C(Rank)[T.", "Rank: ").replace("]", "")
    label = label.replace("post2022:C(Rank)[T.", "post2022 × Rank: ").replace("]", "")
    return {
        "Variable":  label + sig,
        "Coef.":     f"{coef:.4f}",
        "Std. Err.": f"{se:.4f}",
        "t-stat":    f"{t:.3f}",
        "p-value":   f"{p:.4f}",
        "95% CI":    f"[{ci_lo:.4f}, {ci_hi:.4f}]",
    }

for term in priority_terms:
    if term in params_did.index:
        did_rows.append(did_row(term))

did_rows.append({"Variable": "Rank FE",   "Coef.": "Yes", "Std. Err.": "", "t-stat": "", "p-value": "", "95% CI": ""})
did_rows.append({"Variable": "Campus FE", "Coef.": "Yes", "Std. Err.": "", "t-stat": "", "p-value": "", "95% CI": ""})

for term in interaction_terms_did:
    did_rows.append(did_row(term))

did_note = (
    f"N = {int(model_did.nobs):,}   R² = {model_did.rsquared:.4f}\n"
    "DiD spec: log_total_wages ~ post2022 * C(Rank) + C(EmployerName) + Year.\n"
    "HC1 heteroskedasticity-robust standard errors.   Stars: *** p<0.001  ** p<0.01  * p<0.05\n"
    "Reference category: Assistant professors."
)

render_regression_table(
    rows=did_rows,
    col_headers=["Variable", "Coef.", "Std. Err.", "t-stat", "p-value", "95% CI"],
    title="Table 3. DiD Compression: log(Total Wages) ~ post2022 × C(Rank) + Campus FE + Year",
    note=did_note,
    save_name="table_did_compression.png",
    col_widths=REG_COL_WIDTHS,
)

# ══════════════════════════════════════════════════════════════════════════════
# TABLES 4–8: Within-rank regressions
# ══════════════════════════════════════════════════════════════════════════════

rank_file_map = {
    "Assistant": "table_within_rank_assistant.png",
    "Associate": "table_within_rank_associate.png",
    "Full":      "table_within_rank_full.png",
    "Clinical":  "table_within_rank_clinical.png",
    "Research":  "table_within_rank_research.png",
}

for rank, filename in rank_file_map.items():
    if rank not in rank_models:
        print(f"  Skipping {filename} (insufficient data)")
        continue
    m = rank_models[rank]
    params_r = m.params
    bse_r    = m.bse
    tvals_r  = m.tvalues
    pvals_r  = m.pvalues
    ci_r     = m.conf_int()

    rank_rows = []
    campus_shown = False
    for term in params_r.index:
        if "employername" in term.lower():
            if not campus_shown:
                rank_rows.append({
                    "Variable":  "Campus FE",
                    "Coef.":     "Yes",
                    "Std. Err.": "", "t-stat": "", "p-value": "", "95% CI": "",
                })
                campus_shown = True
            continue
        coef  = params_r[term]
        se    = bse_r[term]
        t     = tvals_r[term]
        p     = pvals_r[term]
        sig   = stars(p)
        ci_lo = ci_r.loc[term, 0]
        ci_hi = ci_r.loc[term, 1]
        rank_rows.append({
            "Variable":  term + sig,
            "Coef.":     f"{coef:.4f}",
            "Std. Err.": f"{se:.4f}",
            "t-stat":    f"{t:.3f}",
            "p-value":   f"{p:.4f}",
            "95% CI":    f"[{ci_lo:.4f}, {ci_hi:.4f}]",
        })

    rank_note = (
        f"N = {int(m.nobs):,}   R² = {m.rsquared:.4f}\n"
        f"Formula: log_total_wages ~ post2022 + C(EmployerName) + Year  |  {rank} professors only.\n"
        "HC1 heteroskedasticity-robust standard errors.   Stars: *** p<0.001  ** p<0.01  * p<0.05"
    )

    render_regression_table(
        rows=rank_rows,
        col_headers=["Variable", "Coef.", "Std. Err.", "t-stat", "p-value", "95% CI"],
        title=f"Within-Rank Regression: {rank} Professors\nlog(Total Wages) ~ post2022 + Campus FE + Year",
        note=rank_note,
        save_name=filename,
        col_widths=[3.2, 1, 1, 1, 1, 1.8],
    )

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 9: table_levene_test.png
# ══════════════════════════════════════════════════════════════════════════════

def fmt_large(x):
    if abs(x) >= 1e9:
        return f"{x:.3e}"
    if abs(x) >= 1e6:
        return f"{x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.1f}K"
    return f"{x:.2f}"

levene_rows_table = []
for _, row in levene_df.iterrows():
    levene_rows_table.append({
        "Rank":               row["Rank"],
        "Pre-2022 Variance":  fmt_large(row["Pre-2022 Variance"]),
        "Post-2022 Variance": fmt_large(row["Post-2022 Variance"]),
        "Variance Ratio":     f"{row['Variance Ratio']:.4f}",
        "Levene Stat":        f"{row['Levene Stat']:.4f}",
        "p-value":            f"{row['p-value']:.4e}",
        "Significant":        row["Significant"],
    })

levene_note = (
    "Levene's test for equality of variances (pre-2022 vs post-2022) within each professor rank.\n"
    "Variance Ratio > 1: variance increased post-2022.   Significant at α = 0.05."
)

render_regression_table(
    rows=levene_rows_table,
    col_headers=["Rank", "Pre-2022 Variance", "Post-2022 Variance",
                 "Variance Ratio", "Levene Stat", "p-value", "Significant"],
    title="Table: Levene's Test for Variance Equality — Pre-2022 vs Post-2022",
    note=levene_note,
    save_name="table_levene_test.png",
    col_widths=[1.2, 1.6, 1.6, 1.4, 1.2, 1.2, 1.0],
)

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 10: table_vif.png
# ══════════════════════════════════════════════════════════════════════════════

vif_table_rows = []
for _, row in vif_df.iterrows():
    flag = ""
    if row["VIF"] > 10:
        flag = " [severe]"
    elif row["VIF"] > 5:
        flag = " [moderate]"
    vif_table_rows.append({
        "Predictor": row["Predictor"] + flag,
        "VIF":       f"{row['VIF']:.3f}",
    })

vif_note = (
    "VIF computed using label-encoded Rank and EmployerName (approximation).\n"
    "Threshold: VIF > 5 = moderate concern; VIF > 10 = severe multicollinearity.\n"
    "Model: log_total_wages ~ post2022 + C(Rank) + C(EmployerName) + Year  |  HC1 robust SE."
)

render_regression_table(
    rows=vif_table_rows,
    col_headers=["Predictor", "VIF"],
    title="Table: Variance Inflation Factors (VIF) — Baseline Regression Predictors",
    note=vif_note,
    save_name="table_vif.png",
    col_widths=[2.5, 1.0],
)

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 11: table_cv_by_rank_year.png
# ══════════════════════════════════════════════════════════════════════════════

cv_rows_table = []
for year_idx, year_row in cv_pivot.iterrows():
    row_dict = {"Year": str(int(year_idx))}
    for rank in RANKS:
        val = year_row.get(rank, np.nan)
        row_dict[rank] = f"{val:.2f}" if not np.isnan(val) else "—"
    cv_rows_table.append(row_dict)

cv_hdrs = ["Year"] + RANKS

cv_note = (
    "Coefficient of Variation (CV%) = 100 × (Std Dev / Mean).  Computed from deduplicated panel.\n"
    "Sample: professors with TotalWages > 0, years 2012–2024."
)

render_regression_table(
    rows=cv_rows_table,
    col_headers=cv_hdrs,
    title="Table: Coefficient of Variation (CV%) by Rank and Year",
    note=cv_note,
    save_name="table_cv_by_rank_year.png",
    col_widths=[0.8, 1.1, 1.1, 1.0, 1.1, 1.1],
)

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════

print("\nAll files saved to:", FIG_DIR)
