"""
Parse 14 Open Finance API call Excel files (weekly data from OF Dashboard)
and aggregate to monthly totals.

Input:  data/Chamada de APIs (1).xlsx  through  data/Chamada de APIs (14).xlsx
Output: data/of_api_monthly.csv

API category mapping:
  1  Adiantamento a Depositantes
  2  Canais de Atendimentos
  3  Cartao de Credito              [credit-relevant]
  4  Contas                         [credit-relevant]
  5  Credenciamento
  6  Cambio
  7  Direitos Creditorios Descontados [credit-relevant]
  8  Emprestimos                    [credit-relevant]
  9  Financiamentos                 [credit-relevant]
  10 Investimentos
  11 Previdencia
  12 Produtos e Servicos
  13 Seguros
  14 Titulos de Capitalizacao
"""

import pandas as pd
from pathlib import Path

API_NAMES = {
    1: "adiantamento_depositantes",
    2: "canais_atendimentos",
    3: "cartao_credito",
    4: "contas",
    5: "credenciamento",
    6: "cambio",
    7: "direitos_creditorios",
    8: "emprestimos",
    9: "financiamentos",
    10: "investimentos",
    11: "previdencia",
    12: "produtos_servicos",
    13: "seguros",
    14: "titulos_capitalizacao",
}

CREDIT_CATEGORIES = [3, 7, 8, 9]

data_dir = Path("data")
all_monthly = {}

for i in range(1, 15):
    fpath = data_dir / f"Chamada de APIs ({i}).xlsx"
    col_name = f"api_{API_NAMES[i]}"

    # Data starts at row 11 (0-indexed: skiprows=9 gets us to row 10 header)
    df = pd.read_excel(fpath, sheet_name=0, skiprows=9)
    df = df.rename(columns={"_id": "id", "total": "total", "date": "date"})
    df = df.dropna(subset=["date"])

    # Parse dates and aggregate to monthly
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")["total"].sum().reset_index()
    monthly.columns = ["month", col_name]

    all_monthly[i] = monthly
    print(f"  {i:2d}. {API_NAMES[i]:35s} | {len(df):3d} weeks -> {len(monthly):2d} months")

# Merge all categories
result = all_monthly[1]
for i in range(2, 15):
    result = result.merge(all_monthly[i], on="month", how="outer")

# Sort by month
result = result.sort_values("month").reset_index(drop=True)

# Create aggregate columns
api_cols = [f"api_{API_NAMES[i]}" for i in range(1, 15)]
credit_cols = [f"api_{API_NAMES[i]}" for i in CREDIT_CATEGORIES]

result["api_total"] = result[api_cols].sum(axis=1)
result["api_credit"] = result[credit_cols].sum(axis=1)

# Convert period to end-of-month date
result["month"] = result["month"].dt.to_timestamp("M")

# Write output
outpath = data_dir / "of_api_monthly.csv"
result.to_csv(outpath, index=False)

print(f"\nOutput: {outpath}")
print(f"  {len(result)} months, {result['month'].min().date()} to {result['month'].max().date()}")
print(f"  Columns: {list(result.columns)}")
print(f"\nMonthly totals (last 6 months):")
print(result[["month", "api_total", "api_credit", "api_emprestimos"]].tail(6).to_string(index=False))
