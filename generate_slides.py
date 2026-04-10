"""
Generate presentation-ready graphs for an executive summary:
  Slide 1: OF Customers vs Credit Portfolio (dual-axis) + Default Rate
  Slide 2: Coefficient plot — Credit Growth (stock) on ln(Customers)
  Slide 3: Coefficient plot — Default Rate Change on ln(Customers)
  Slide 4: Coefficient plot — Credit Origination Growth on ln(Customers)
  Slide 5: Coefficient plot — SME Credit & Interest Rates on ln(Customers)

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

# ── Load BCB SGS data (originations, SME, rates) ────────────────────────

def load_sgs_csv(filepath, colname):
    """Load a BCB SGS CSV (semicolon-sep, DD/MM/YYYY dates, comma decimals)."""
    try:
        raw = pd.read_csv(filepath, sep=";", encoding="utf-8")
        raw.columns = ["date_str", "value"]
        raw["month"] = pd.to_datetime(raw["date_str"], format="%d/%m/%Y") + pd.offsets.MonthEnd(0)
        raw[colname] = raw["value"].astype(str).str.replace(",", ".").astype(float)
        return raw[["month", colname]]
    except Exception as e:
        print(f"  Could not load {filepath}: {e}")
        return None

sgs_files = {
    "orig_total":          "data/concessoes_total.csv",
    "orig_pf_personal_nc": "data/concessoes_pf_pessoal_nao_consig.csv",
    "orig_pj_total":       "data/concessoes_pj_total.csv",
    "orig_pj_working_cap": "data/concessoes_pj_capital_giro.csv",
    "credit_mpme":         "data/credito_mpme.csv",
    "rate_pf_personal_nc": "data/taxa_juros_pf_pessoal_nao_consig.csv",
    "rate_pj_small":       "data/taxa_juros_pj_pequeno_porte.csv",
}

for colname, fpath in sgs_files.items():
    sgs_df = load_sgs_csv(fpath, colname)
    if sgs_df is not None:
        df = df.merge(sgs_df, on="month", how="left")
        print(f"  Loaded {colname} ({len(sgs_df)} rows)")

# ── Load OF API call data ────────────────────────────────────────────────
api_path = Path("data/of_api_monthly.csv")
if api_path.exists():
    of_api = pd.read_csv(api_path)
    of_api["month"] = pd.to_datetime(of_api["month"]) + pd.offsets.MonthEnd(0)
    df = df.merge(of_api, on="month", how="left")
    print(f"  Loaded OF API data ({len(of_api)} months)")

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

# Origination growth rates
for col in ["orig_total", "orig_pf_personal_nc", "orig_pj_total", "orig_pj_working_cap"]:
    if col in df.columns:
        ln_col = f"ln_{col}"
        df[ln_col] = safe_log(df[col])
        df[f"g_yoy_{col}"] = 100 * (df[ln_col] - df[ln_col].shift(12))

# SME credit growth
if "credit_mpme" in df.columns:
    df["ln_credit_mpme"] = safe_log(df["credit_mpme"])
    df["g_yoy_credit_mpme"] = 100 * (df["ln_credit_mpme"] - df["ln_credit_mpme"].shift(12))

# Interest rate YoY changes
for col in ["rate_pf_personal_nc", "rate_pj_small"]:
    if col in df.columns:
        df[f"d_{col}_yoy"] = df[col] - df[col].shift(12)

# API call treatment variable
if "api_credit" in df.columns:
    df["ln_api_credit"] = safe_log(df["api_credit"])


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

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Coefficient Plot: Credit Origination Growth (Flow-Based)
# ══════════════════════════════════════════════════════════════════════════

orig_models = {}
orig_spec = {
    "Total\nOriginations":      "g_yoy_orig_total",
    "PF Personal\n(Non-Payroll)": "g_yoy_orig_pf_personal_nc",
    "PJ Total\nOriginations":   "g_yoy_orig_pj_total",
    "PJ Working\nCapital":      "g_yoy_orig_pj_working_cap",
}

for label, y_col in orig_spec.items():
    if y_col in df.columns and df[y_col].notna().sum() >= 10:
        try:
            res = run_ols_nw(y_col, controls + ["ln_cust"], df)
            orig_models[label] = {
                "beta": res["params"]["ln_cust"],
                "se":   res["bse"]["ln_cust"],
                "p":    res["pvalues"]["ln_cust"],
                "n":    res["nobs"],
                "r2":   res["r2"],
            }
        except Exception as e:
            print(f"  Skipping {label}: {e}")

if orig_models:
    coef_plot(
        orig_models,
        title="Effect of Open Finance on Credit Origination Growth (YoY)",
        xlabel="Coefficient on ln(Customers)",
        filename="slide4_origination_coefs.png",
        positive_good=True,
    )
else:
    print("Skipped slide 4 (no origination data). Run download_bcb_data.R first.")


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Coefficient Plot: SME Credit & Interest Rate Models
# ══════════════════════════════════════════════════════════════════════════

inclusion_models = {}

# SME credit growth
if "g_yoy_credit_mpme" in df.columns and df["g_yoy_credit_mpme"].notna().sum() >= 10:
    try:
        res = run_ols_nw("g_yoy_credit_mpme", controls + ["ln_cust"], df)
        inclusion_models["SME Credit\nGrowth (YoY)"] = {
            "beta": res["params"]["ln_cust"],
            "se":   res["bse"]["ln_cust"],
            "p":    res["pvalues"]["ln_cust"],
            "n":    res["nobs"],
            "r2":   res["r2"],
        }
    except Exception as e:
        print(f"  Skipping SME model: {e}")

# Interest rate changes
rate_spec = {
    "PF Personal Rate\nChange (YoY pp)": "d_rate_pf_personal_nc_yoy",
    "PJ Small Biz Rate\nChange (YoY pp)": "d_rate_pj_small_yoy",
}

for label, y_col in rate_spec.items():
    if y_col in df.columns and df[y_col].notna().sum() >= 10:
        try:
            res = run_ols_nw(y_col, controls + ["ln_cust"], df)
            inclusion_models[label] = {
                "beta": res["params"]["ln_cust"],
                "se":   res["bse"]["ln_cust"],
                "p":    res["pvalues"]["ln_cust"],
                "n":    res["nobs"],
                "r2":   res["r2"],
            }
        except Exception as e:
            print(f"  Skipping {label}: {e}")

if inclusion_models:
    coef_plot(
        inclusion_models,
        title="Effect of Open Finance on SME Credit & Interest Rates",
        xlabel="Coefficient on ln(Customers)",
        filename="slide5_sme_rates_coefs.png",
        positive_good=True,  # For SME growth positive is good; for rates negative is good
    )
else:
    print("Skipped slide 5 (no SME/rate data). Run download_bcb_data.R first.")

# ══════════════════════════════════════════════════════════════════════════
# SLIDE 6 — API Calls Time Series (Credit vs Other)
# ══════════════════════════════════════════════════════════════════════════

has_api_slide = False
if "api_credit" in df.columns:
    api_mask = df["api_total"].notna()
    if api_mask.sum() > 5:
        fig, ax = plt.subplots(figsize=(11, 5.5))

        other = df.loc[api_mask, "api_total"] - df.loc[api_mask, "api_credit"]
        ax.fill_between(df.loc[api_mask, "month"], 0, other / 1e3,
                        alpha=0.4, color=GREY, label="Other APIs")
        ax.fill_between(df.loc[api_mask, "month"], other / 1e3,
                        (other + df.loc[api_mask, "api_credit"]) / 1e3,
                        alpha=0.7, color=BLUE, label="Credit-Related APIs")

        ax.set_title("Open Finance API Calls by Category (Monthly)", pad=12)
        ax.set_ylabel("API Calls (thousands)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.legend(loc="upper left")

        fig.tight_layout()
        fig.savefig(OUT / "slide6_api_calls_timeseries.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Saved slide6_api_calls_timeseries.png")
        has_api_slide = True


# ══════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Coefficient Plot: API Credit Calls as Treatment Variable
# ══════════════════════════════════════════════════════════════════════════

has_api_coef_slide = False
if "ln_api_credit" in df.columns:
    api_models = {}
    api_spec = {
        "Total Credit\nGrowth (YoY)":    ("g_yoy_credit",      controls + ["ln_api_credit"]),
        "PF Personal\nLoans (YoY)":      ("g_yoy_pf_personal", controls + ["ln_api_credit"]),
        "Default Rate\nChange (YoY)":    ("d_default_yoy",     controls + ["ln_api_credit"]),
    }

    for label, (y_col, x_cols) in api_spec.items():
        if y_col in df.columns and df[y_col].notna().sum() >= 10:
            try:
                res = run_ols_nw(y_col, x_cols, df)
                api_models[label] = {
                    "beta": res["params"]["ln_api_credit"],
                    "se":   res["bse"]["ln_api_credit"],
                    "p":    res["pvalues"]["ln_api_credit"],
                    "n":    res["nobs"],
                    "r2":   res["r2"],
                }
            except Exception as e:
                print(f"  Skipping API model {label}: {e}")

    if api_models:
        coef_plot(
            api_models,
            title="Effect of OF Credit API Calls on Credit Outcomes",
            xlabel="Coefficient on ln(API Credit Calls)",
            filename="slide7_api_credit_coefs.png",
            positive_good=True,
        )
        has_api_coef_slide = True
    else:
        print("Skipped slide 7 (insufficient API data for regressions)")

n_slides = (4 + (1 if orig_models else 0) + (1 if inclusion_models else 0)
            + (1 if has_api_slide else 0) + (1 if has_api_coef_slide else 0))
print(f"\nAll {n_slides} graphs saved to slides/")
