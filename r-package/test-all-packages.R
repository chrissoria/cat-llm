#!/usr/bin/env Rscript
# ============================================================================
# Smoke-test all 8 R packages in the CatLLM ecosystem.
#
# Usage:
#   OPENAI_API_KEY=sk-...  Rscript r-package/test-all-packages.R
#
# Optional:
#   CATLLM_R_ROOT=/path/to/r-package    # override path discovery
#   CATLLM_SKIP_INSTALL=1               # skip R/Python install steps
#
# Each per-package test runs a minimal classify() (or analogous) call on a
# tiny dataset against OpenAI gpt-4o-mini. Total cost is a few cents.
# ============================================================================

suppressPackageStartupMessages({
  library(devtools)
  library(reticulate)
})

`%||%` <- function(a, b) if (is.null(a)) b else a

# ---- Path discovery -------------------------------------------------------
ROOT <- Sys.getenv("CATLLM_R_ROOT", "")
if (!nzchar(ROOT)) {
  script_path <- tryCatch(
    normalizePath(sys.frame(1)$ofile, mustWork = TRUE),
    error = function(e) NULL
  )
  ROOT <- if (!is.null(script_path)) dirname(script_path) else "r-package"
}
ROOT <- normalizePath(ROOT, mustWork = TRUE)
message("Using R package root: ", ROOT)

# ---- Config ---------------------------------------------------------------
R_PKGS  <- c("cat.stack", "cat.survey", "cat.vader", "cat.ademic",
             "cat.cog",   "cat.pol",    "cat.web",   "cat.llm")

# cat.llm has no Python counterpart; cat.stack is installed via cat-stack on pip.
PY_PKGS <- c("cat-stack", "cat-survey", "cat-vader", "cat-ademic",
             "cat-cog",   "cat-pol",    "cat-web")

API_KEY <- Sys.getenv("OPENAI_API_KEY", "")
if (!nzchar(API_KEY)) {
  stop("OPENAI_API_KEY environment variable is not set.", call. = FALSE)
}
MODEL <- "gpt-4o-mini"

SKIP_INSTALL <- nzchar(Sys.getenv("CATLLM_SKIP_INSTALL", ""))

# ---- Results bookkeeping --------------------------------------------------
results <- data.frame(
  package = character(), status = character(),
  detail = character(), elapsed_s = numeric(),
  stringsAsFactors = FALSE
)
record <- function(pkg, status, detail = "", t0 = NULL) {
  elapsed <- if (is.null(t0)) NA_real_
             else round(as.numeric(Sys.time() - t0, units = "secs"), 2)
  results[nrow(results) + 1L, ] <<- list(pkg, status, detail, elapsed)
  cat(sprintf("  [%s] %s  %s\n", status, pkg, detail))
}

run <- function(pkg, expr) {
  t0 <- Sys.time()
  tryCatch({
    val <- eval(expr, envir = parent.frame())
    detail <- paste0("class=", class(val)[1L],
                     " nrow=", tryCatch(NROW(val), error = function(e) NA_integer_))
    record(pkg, "PASS", detail, t0)
  }, error = function(e) {
    record(pkg, "FAIL_RUN", conditionMessage(e), t0)
  })
}

# ---- Step 1: install R packages from local source (dependency order) -----
if (!SKIP_INSTALL) {
  cat("\n============ Installing R packages ============\n")
  for (pkg in R_PKGS) {
    path <- file.path(ROOT, pkg)
    cat("install_local: ", path, "\n", sep = "")
    tryCatch(
      devtools::install_local(
        path, force = TRUE, upgrade = "never",
        quiet = TRUE, dependencies = FALSE
      ),
      error = function(e) {
        record(pkg, "FAIL_INSTALL_R", conditionMessage(e))
      }
    )
  }
}

# ---- Step 2: install Python dependencies ---------------------------------
if (!SKIP_INSTALL) {
  cat("\n============ Installing Python deps ============\n")
  for (py in PY_PKGS) {
    cat("py_install: ", py, "\n", sep = "")
    try(reticulate::py_install(py, pip = TRUE), silent = FALSE)
  }
}

# ---- Step 3: per-package smoke tests --------------------------------------
cat("\n============ Smoke tests ============\n")

sample_text <- c(
  "The bill allocates $50M for affordable housing in fiscal year 2026.",
  "Just had the best coffee ever! Highly recommend the new place downtown.",
  "We thank the anonymous reviewers for their thoughtful feedback."
)

# --- cat.stack ---
run("cat.stack", quote(cat.stack::classify(
  categories      = c("policy", "consumer", "academic"),
  input_data      = sample_text,
  api_key         = API_KEY,
  user_model      = MODEL,
  check_verbosity = FALSE
)))

# --- cat.survey ---
run("cat.survey", quote(cat.survey::classify(
  categories      = c("policy", "consumer", "academic"),
  input_data      = sample_text,
  survey_question = "What is this respondent talking about?",
  api_key         = API_KEY,
  user_model      = MODEL,
  check_verbosity = FALSE
)))

# --- cat.vader (text-only path: no sm_source/sm_handle/sm_credentials) ---
run("cat.vader", quote(cat.vader::classify(
  categories      = c("positive", "negative", "neutral"),
  input_data      = sample_text,
  api_key         = API_KEY,
  user_model      = MODEL,
  check_verbosity = FALSE
)))

# --- cat.ademic (mode="text" to skip image processing) ---
run("cat.ademic", quote(cat.ademic::classify(
  categories      = c("methods", "results", "other"),
  input_data      = sample_text,
  api_key         = API_KEY,
  user_model      = MODEL,
  mode            = "text",
  check_verbosity = FALSE
)))

# --- cat.pol (text path, no `source` arg) ---
run("cat.pol", quote(cat.pol::classify(
  categories      = c("housing", "public safety", "finance"),
  input_data      = sample_text,
  api_key         = API_KEY,
  user_model      = MODEL,
  check_verbosity = FALSE
)))

# --- cat.web (text path, no source_domain/content_type) ---
run("cat.web", quote(cat.web::classify(
  categories      = c("news", "opinion", "tutorial"),
  input_data      = sample_text,
  api_key         = API_KEY,
  user_model      = MODEL,
  check_verbosity = FALSE
)))

# --- cat.cog (generate a tiny PNG circle as test image) ---
img_path <- tryCatch({
  tmp <- tempfile(fileext = ".png")
  grDevices::png(tmp, width = 200, height = 200, bg = "white")
  graphics::par(mar = c(0, 0, 0, 0))
  graphics::plot.new()
  graphics::plot.window(xlim = c(0, 1), ylim = c(0, 1), asp = 1)
  graphics::symbols(0.5, 0.5, circles = 0.4, inches = FALSE,
                    add = TRUE, fg = "black", lwd = 3)
  grDevices::dev.off()
  tmp
}, error = function(e) NULL)

if (is.null(img_path) || !file.exists(img_path)) {
  record("cat.cog", "SKIP", "could not generate test PNG")
} else {
  run("cat.cog", bquote(cat.cog::cerad_drawn_score(
    shape       = "circle",
    image_input = .(img_path),
    api_key     = API_KEY,
    user_model  = MODEL
  )))
}

# --- cat.llm meta: verify it loads + aliases work + one round-trip ---
run("cat.llm", quote({
  suppressPackageStartupMessages(library(cat.llm))
  stopifnot(
    is.function(classify_survey),
    is.function(classify_political),
    is.function(classify_web),
    is.function(cerad_drawn_score)
  )
  cat.llm::classify_survey(
    categories      = c("policy", "consumer", "academic"),
    input_data      = sample_text[1:2],
    survey_question = "What is this respondent talking about?",
    api_key         = API_KEY,
    user_model      = MODEL,
    check_verbosity = FALSE
  )
}))

# ---- Step 4: final summary -----------------------------------------------
cat("\n================ SMOKE TEST SUMMARY ================\n")
print(results, row.names = FALSE)
n_pass <- sum(results$status == "PASS")
n_fail <- sum(grepl("^FAIL", results$status))
n_skip <- sum(results$status == "SKIP")
cat(sprintf("\n%d / %d passed  (%d failed, %d skipped)\n",
            n_pass, nrow(results), n_fail, n_skip))
quit(status = if (n_fail > 0L) 1L else 0L)
