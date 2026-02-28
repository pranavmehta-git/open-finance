"""
Generate 4 presentation-ready graphs for a 3-slide executive summary:
  Slide 1: OF Customers vs Credit Portfolio (dual-axis) + Default Rate
  Slide 2: Coefficient plot — Credit Growth on ln(Customers)
  Slide 3: Coefficient plot — Default Rate Change on ln(Customers)

Outputs saved to slides/ as high-res PNGs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import statsmodels.api as sm
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

PURPLE = "#5B2C87"
RED = "#C0392B"
GREEN = "#1E8449"
BLUE = "#2471A3"
ORANGE = "#E67E22"
GREY = "#7F8C8D"

OUT = Path("slides")
OUT.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────

def end_of_month(s):
    """Parse 'MM/YYYY' string to end-of-month Timestamp."""
    dt = pd.to_datetime(s, format="%m/%Y")
    return dt + pd.offsets.MonthEnd(0)

# PIX / macro / customers
df_long = pd.read_excel("data/data_monthly.xlsx", sheet_name="data2")
df_long["month"] = df_long["month"].apply(end_of_month)

df_short = pd.read_excel("data/data_monthly.xlsx", sheet_name="data1")
df_short["month"] = df_short["month"].apply(end_of_month)

df = df_long.merge(df_short[["month", "cust"]], on="month", how="left")

# SCR national totals
cart = pd.read_csv("data/carteira.csv", encoding="utf-8-sig")
cart.columns = ["date", "credit_portfolio", "_drop"]
cart = cart[["date", "credit_portfolio"]]

inad = pd.read_csv("data/inadimplncia.csv", encoding="utf-8-sig")
inad.columns = ["date", "default_rate", "_drop"]
inad = inad[["date", "default_rate"]]

scr = cart.merge(inad, on="date")
scr["month"] = pd.to_datetime(scr["date"] + "-01") + pd.offsets.MonthEnd(0)
scr = scr.drop(columns="date")

# SCR PF/PJ
cart_pf_pj = pd.read_csv("data/carteira_pf_pj.csv", encoding="utf-8-sig")
cart_pf_pj.columns = ["date", "_t1", "_t2", "credit_pf", "credit_pj"]
cart_pf_pj = cart_pf_pj[["date", "credit_pf", "credit_pj"]]

inad_pf_pj = pd.read_csv("data/inadimplencia_pf_pj.csv", encoding="utf-8-sig")
inad_pf_pj.columns = ["date", "_t1", "_t2", "default_rate_pf", "default_rate_pj"]
inad_pf_pj = inad_pf_pj[["date", "default_rate_pf", "default_rate_pj"]]

scr_pf_pj = cart_pf_pj.merge(inad_pf_pj, on="date")
scr_pf_pj["month"] = pd.to_datetime(scr_pf_pj["date"] + "-01") + pd.offsets.MonthEnd(0)
scr_pf_pj = scr_pf_pj.drop(columns="date")

# SCR PF modalities (personal loans = col index 7, i.e. column 8 in 1-indexed)
cart_mod = pd.read_csv("data/carteira_pf_modalities.csv", encoding="utf-8-sig")
cart_mod = cart_mod.iloc[:, [0, 5, 7]]  # date, card (col6), personal (col8)
cart_mod.columns = ["date", "credit_pf_card", "credit_pf_personal"]
cart_mod["month"] = pd.to_datetime(cart_mod["date"] + "-01") + pd.offsets.MonthEnd(0)
cart_mod = cart_mod.drop(columns="date")

# ── Merge ─────────────────────────────────────────────────────────────────

df = (
    df.merge(scr, on="month", how="left")
      .merge(scr_pf_pj, on="month", how="left")
      .merge(cart_mod, on="month", how="left")
      .sort_values("month")
      .reset_index(drop=True)
)

# ── Features ──────────────────────────────────────────────────────────────

def safe_log(s):
    return np.where((s.notna()) & (s > 0), np.log(s), np.nan)

df["ln_cust"] = safe_log(df["cust"])
df["ln_credit"] = safe_log(df["credit_portfolio"])
df["ln_ibc_br"] = np.log(df["ibc_br"])
df["ln_credit_pf"] = safe_log(df["credit_pf"])
df["ln_credit_pj"] = safe_log(df["credit_pj"])
df["ln_pf_personal"] = safe_log(df["credit_pf_personal"])

df["g_yoy_credit"]      = 100 * (df["ln_credit"] - df["ln_credit"].shift(12))
df["g_yoy_credit_pf"]   = 100 * (df["ln_credit_pf"] - df["ln_credit_pf"].shift(12))
df["g_yoy_credit_pj"]   = 100 * (df["ln_credit_pj"] - df["ln_credit_pj"].shift(12))
df["g_yoy_pf_personal"] = 100 * (df["ln_pf_personal"] - df["ln_pf_personal"].shift(12))
df["g_yoy_ibc_br"]      = 100 * (df["ln_ibc_br"] - df["ln_ibc_br"].shift(12))

df["d_default_yoy"]    = df["default_rate"] - df["default_rate"].shift(12)
df["d_default_pf_yoy"] = df["default_rate_pf"] - df["default_rate_pf"].shift(12)
df["d_default_pj_yoy"] = df["default_rate_pj"] - df["default_rate_pj"].shift(12)


# ── Regression helper ─────────────────────────────────────────────────────

def run_ols_nw(y_col, x_cols, data):
    """OLS with Newey-West SEs. Returns dict with named params/bse/pvalues."""
    sub = data[["month", y_col] + x_cols].dropna()
    Y = sub[y_col]
    X = sm.add_constant(sub[x_cols])
    model = sm.OLS(Y, X).fit()
    nw_lag = max(1, int(0.75 * len(sub) ** (1 / 3)))
    nw = model.get_robustcov_results(cov_type="HAC", maxlags=nw_lag, use_correction=True)
    names = list(X.columns)
    return {
        "params":  dict(zip(names, nw.params)),
        "bse":     dict(zip(names, nw.bse)),
        "pvalues": dict(zip(names, nw.pvalues)),
        "nobs":    int(nw.nobs),
        "r2":      nw.rsquared,
    }


# ── Run regressions ───────────────────────────────────────────────────────

controls = ["selic", "g_yoy_ibc_br"]

credit_models = {
    "Total Credit":     ("g_yoy_credit",      controls + ["ln_cust"]),
    "PF (Individuals)": ("g_yoy_credit_pf",   controls + ["ln_cust"]),
    "PJ (Firms)":       ("g_yoy_credit_pj",   controls + ["ln_cust"]),
    "PF Personal\nLoans": ("g_yoy_pf_personal", controls + ["ln_cust"]),
}

default_models = {
    "Aggregate":        ("d_default_yoy",    controls + ["ln_cust"]),
    "PF (Individuals)": ("d_default_pf_yoy", controls + ["ln_cust"]),
    "PJ (Firms)":       ("d_default_pj_yoy", controls + ["ln_cust"]),
}

credit_results = {}
for label, (y, x) in credit_models.items():
    res = run_ols_nw(y, x, df)
    credit_results[label] = {
        "beta": res["params"]["ln_cust"],
        "se":   res["bse"]["ln_cust"],
        "p":    res["pvalues"]["ln_cust"],
        "n":    res["nobs"],
        "r2":   res["r2"],
    }

default_results = {}
for label, (y, x) in default_models.items():
    res = run_ols_nw(y, x, df)
    default_results[label] = {
        "beta": res["params"]["ln_cust"],
        "se":   res["bse"]["ln_cust"],
        "p":    res["pvalues"]["ln_cust"],
        "n":    res["nobs"],
        "r2":   res["r2"],
    }


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 1, GRAPH 1 — Dual-axis: Credit Portfolio + OF Customers
# ══════════════════════════════════════════════════════════════════════════

fig, ax1 = plt.subplots(figsize=(11, 5.5))

# Credit portfolio on left axis
mask_cr = df["credit_portfolio"].notna()
ax1.plot(df.loc[mask_cr, "month"], df.loc[mask_cr, "credit_portfolio"] / 1e12,
         color=PURPLE, linewidth=2, label="Credit Portfolio (LHS)")
ax1.set_ylabel("Active Credit Portfolio (R$ Trillions)", color=PURPLE, fontsize=12)
ax1.tick_params(axis="y", labelcolor=PURPLE)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

# OF customers on right axis
ax2 = ax1.twinx()
mask_cu = df["cust"].notna()
ax2.plot(df.loc[mask_cu, "month"], df.loc[mask_cu, "cust"] / 1e6,
         color=GREEN, linewidth=2.5, linestyle="--", marker="o", markersize=4,
         label="OF Customers (RHS)")
ax2.set_ylabel("Open Finance Customers (Millions)", color=GREEN, fontsize=12)
ax2.tick_params(axis="y", labelcolor=GREEN)

# Shade OF era
of_start = pd.Timestamp("2023-01-01")
ax1.axvspan(of_start, df["month"].max(), alpha=0.07, color=GREEN)
ax1.axvline(of_start, color=GREEN, linewidth=0.8, linestyle=":", alpha=0.6)
ax1.text(of_start + pd.Timedelta(days=60), ax1.get_ylim()[1] * 0.97,
         "Open Finance\ndata available", fontsize=9, color=GREEN,
         va="top", fontstyle="italic")

ax1.set_title("Open Finance Rollout & Total Credit Portfolio", pad=12)
ax1.set_xlabel(None)
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT / "slide1_credit_vs_customers.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved slide1_credit_vs_customers.png")


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 1, GRAPH 2 — Default Rate Over Time
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(11, 4.5))

mask_def = df["default_rate"].notna()
ax.plot(df.loc[mask_def, "month"], df.loc[mask_def, "default_rate"],
        color=RED, linewidth=2)

ax.axvspan(of_start, df["month"].max(), alpha=0.07, color=GREEN)
ax.axvline(of_start, color=GREEN, linewidth=0.8, linestyle=":", alpha=0.6)
ax.text(of_start + pd.Timedelta(days=60), ax.get_ylim()[1] * 0.97,
        "Open Finance\ndata available", fontsize=9, color=GREEN,
        va="top", fontstyle="italic")

ax.set_title("Credit Default Rate (Inadimplência) Over Time", pad=12)
ax.set_ylabel("Default Rate (%)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

fig.tight_layout()
fig.savefig(OUT / "slide1_default_rate.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved slide1_default_rate.png")


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Coefficient Plot: Credit Growth
# ══════════════════════════════════════════════════════════════════════════

def coef_plot(results, title, xlabel, filename, positive_good=True):
    labels = list(results.keys())
    betas  = [results[k]["beta"] for k in labels]
    ses    = [results[k]["se"]   for k in labels]
    pvals  = [results[k]["p"]    for k in labels]
    ns     = [results[k]["n"]    for k in labels]
    r2s    = [results[k]["r2"]   for k in labels]

    fig, ax = plt.subplots(figsize=(10, max(4.5, len(labels) * 1.5)))
    y_pos = np.arange(len(labels))

    colors = []
    for b, p in zip(betas, pvals):
        if p > 0.10:
            colors.append(GREY)
        elif (positive_good and b > 0) or (not positive_good and b < 0):
            colors.append(BLUE)
        else:
            colors.append(ORANGE)

    ax.barh(y_pos, betas, xerr=[1.96 * s for s in ses],
            color=colors, edgecolor="white", height=0.5,
            capsize=5, error_kw={"linewidth": 1.5, "color": "#333"})

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, pad=14)

    # Place annotation ABOVE each bar (vertically offset), not inline
    for i, (b, p, n, r2, se) in enumerate(zip(betas, pvals, ns, r2s, ses)):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        coef_txt = f"{b:+.3f}{stars}"
        stat_txt = f"p={p:.3f}, N={n}, R\u00b2={r2:.2f}"
        # Position: above the bar, at the bar's x-center
        ax.annotate(
            f"{coef_txt}\n({stat_txt})",
            xy=(b, i), xytext=(0, -22),
            textcoords="offset points", ha="center", va="top",
            fontsize=8.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
        )

    # Expand x-limits so whiskers don't clip
    all_tips = [abs(b) + 1.96 * s for b, s in zip(betas, ses)]
    margin = max(all_tips) * 0.4
    xmin = min(min(betas) - max(1.96 * s for s in ses) - margin, -margin)
    xmax = max(max(betas) + max(1.96 * s for s in ses) + margin, margin)
    ax.set_xlim(xmin, xmax)

    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


coef_plot(
    credit_results,
    title="Effect of Open Finance on Credit Growth (YoY)",
    xlabel="Coefficient on ln(Customers)",
    filename="slide2_credit_growth_coefs.png",
    positive_good=True,
)


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Coefficient Plot: Default Rate
# ══════════════════════════════════════════════════════════════════════════

coef_plot(
    default_results,
    title="Effect of Open Finance on Default Rate Change (YoY)",
    xlabel="Coefficient on ln(Customers)",
    filename="slide3_default_rate_coefs.png",
    positive_good=False,
)

print("\nAll 4 graphs saved to slides/")
