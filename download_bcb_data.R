#!/usr/bin/env Rscript
# Download BCB SGS time series for credit originations, SME credit, and interest rates
#
# These series are needed for the client-requested analysis focusing on:
#   1. Credit originations (flow-based measures)
#   2. SME finance (Micro, Pequena e Media enterprises)
#   3. Lending conditions (interest rates)
#
# Data source: Banco Central do Brasil - Sistema Gerenciador de Series Temporais (SGS)
# API docs: https://dadosabertos.bcb.gov.br/

library(httr)
library(readr)

series <- list(
  # Credit originations (concessoes) - flow-based measures
  concessoes_total                  = 20631,  # Total credit originations
  concessoes_pf_pessoal_total      = 20672,  # PF - Personal credit total
  concessoes_pf_pessoal_nao_consig = 20666,  # PF - Personal non-payroll (highest info friction)
  concessoes_pf_cartao             = 20656,  # PF - Credit card originations
  concessoes_pj_total              = 20635,  # PJ - Total originations
  concessoes_pj_capital_giro       = 20642,  # PJ - Working capital originations

  # SME credit stock
  credito_mpme                     = 27701,  # Credit balance - Micro, Small & Medium enterprises

  # Interest rates
  taxa_juros_pf_total              = 20716,  # Avg interest rate - PF total
  taxa_juros_pf_pessoal_nao_consig = 25464,  # Avg monthly rate - PF personal non-payroll
  taxa_juros_pj_pequeno_porte      = 26429   # Avg rate - PJ small businesses
)

download_sgs <- function(series_code, filename) {
  url <- paste0(
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.",
    series_code,
    "/dados?formato=csv"
  )
  outpath <- file.path("data", paste0(filename, ".csv"))

  cat("Downloading series", series_code, "->", outpath, "... ")

  tryCatch({
    resp <- GET(url, timeout(30))
    if (status_code(resp) == 200) {
      writeBin(content(resp, "raw"), outpath)
      # Verify the file has data
      df <- read_csv2(outpath, show_col_types = FALSE)
      cat("OK (", nrow(df), "rows)\n")
    } else {
      cat("FAILED (HTTP", status_code(resp), ")\n")
    }
  }, error = function(e) {
    cat("ERROR:", conditionMessage(e), "\n")
  })
}

cat("Downloading BCB SGS series to data/ directory...\n\n")

for (name in names(series)) {
  download_sgs(series[[name]], name)
}

cat("\nDone. Run open_finance.Rmd to incorporate the new data.\n")
