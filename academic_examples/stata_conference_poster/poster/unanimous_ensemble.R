# =============================================================================
# unanimous_ensemble.R — unanimous-vote ensemble macro F1 for the poster's
# "Unanimous Ensembles" chart (fig4_unanimous.png).
# =============================================================================
# Combines row-level per-model predictions with unanimous-AND logic (a
# category is assigned only if every model in the group predicts 1), then
# scores against the same UCNets gold standard used everywhere else on this
# poster. Reuses an existing, tested gold-alignment/scoring library from an
# unpublished companion benchmark rather than re-implementing gold-matching
# here, so results stay consistent with the numbers behind fig2/fig3.
#
# Requires that companion benchmark's source data, which is not included in
# this repository. Set CATLLM_ENSEMBLE_DATA_ROOT to its local path before
# running (see below for the expected layout).
#
# Run:  CATLLM_ENSEMBLE_DATA_ROOT=/path/to/source Rscript unanimous_ensemble.R
#
# Last run 2026-09-22:
#   frontier-open unanimous ensemble (6 models): 0.8044
#     per-question: a19i 0.8786, a19f 0.9238, e1b 0.6107
#   local unanimous ensemble (4 models):         0.6693
#     per-question: a19i 0.7647, a19f 0.7728, e1b 0.4702
#   All 175/175 gold rows matched exactly for every question/tier (no fuzzy
#   fallback needed).
# =============================================================================

HERE <- Sys.getenv("CATLLM_ENSEMBLE_DATA_ROOT", unset = "")
if (!nzchar(HERE)) {
  stop("Set CATLLM_ENSEMBLE_DATA_ROOT to the companion benchmark's local ",
       "path before running (see header comment).")
}
# Expected layout under HERE:
#   R/scoring_lib.R                       shared gold-standard alignment/scoring
#   data/<date>/<run>/{a19i,a19f,e1b}_ensemble.csv   per-model 0/1 predictions
#   models_registry.csv                   model -> tier -> canonical run
source(file.path(HERE, "R", "scoring_lib.R"))

QUESTIONS <- c("a19i", "a19f", "e1b")

# The two runs that between them cover all 6 frontier-open models (one model's
# canonical run lives apart from the other five -- see models_registry.csv's
# source_run column) and the corrected local-tier run (an earlier local run
# had a data-preparation bug affecting long responses, since fixed; use the
# corrected one, matching the leaderboard/plots pipeline's own convention).
FRONTIER_OPEN_MAIN_RUN <- "phase1_high_open"     # 5 of the 6 frontier-open models
FRONTIER_OPEN_EXTRA_RUN <- "phase1"               # the 6th model's canonical run
FRONTIER_OPEN_EXTRA_MODEL <- "qwen_qwen3_6_35b_a3b"
LOCAL_RUN <- "phase1_small_full"                  # corrected local-tier run
DATA_DATE <- "06_03_26"

unanimous_f1 <- function(matrices_list, truth) {
  # matrices_list: list of n_rows x 6 0/1 matrices, all aligned to the SAME truth row order
  n_models <- length(matrices_list)
  stacked <- array(unlist(matrices_list), dim = c(nrow(truth), ncol(truth), n_models))
  unanimous_pred <- apply(stacked, c(1, 2), function(v) as.integer(all(v == 1)))
  # macro F1, matching scoring_lib.R's compute_metrics(): per-category F1 (NA
  # if no gold support), mean over supported categories.
  f1s <- numeric(ncol(truth))
  for (j in seq_len(ncol(truth))) {
    p <- unanimous_pred[, j]; t <- truth[, j]
    tp <- sum(p == 1 & t == 1); fp <- sum(p == 1 & t == 0); fn <- sum(p == 0 & t == 1)
    f1s[j] <- if (tp + fn == 0) NA_real_ else 2 * tp / (2 * tp + fp + fn)
  }
  mean(f1s, na.rm = TRUE)
}

frontier_open_f1 <- c()
local_f1 <- c()

for (qc in QUESTIONS) {
  message("=== ", qc, " ===")

  # -- frontier-open: the 5-model run plus the 6th model's own canonical run.
  # Both are independently aligned to the same gold standard via
  # align_ensemble(); the 6th model's matrix is then reindexed onto the main
  # run's gold-row order (not raw row position, which is not guaranteed to
  # match across files) before the unanimous-AND combine.
  main <- align_ensemble(file.path(HERE, "data", DATA_DATE, FRONTIER_OPEN_MAIN_RUN), qc)
  extra_run <- align_ensemble(file.path(HERE, "data", DATA_DATE, FRONTIER_OPEN_EXTRA_RUN), qc)

  stopifnot(length(main$matrices) == 5)
  stopifnot(FRONTIER_OPEN_EXTRA_MODEL %in% names(extra_run$matrices))

  extra_by_goldidx <- extra_run$matrices[[FRONTIER_OPEN_EXTRA_MODEL]]
  rownames(extra_by_goldidx) <- extra_run$idx
  extra_aligned <- extra_by_goldidx[as.character(main$idx), ]
  stopifnot(!any(is.na(extra_aligned)))

  fo_matrices <- c(main$matrices, setNames(list(extra_aligned), FRONTIER_OPEN_EXTRA_MODEL))
  stopifnot(length(fo_matrices) == 6)
  f1_fo <- unanimous_f1(fo_matrices, main$truth)
  message("  frontier-open unanimous macro_f1 = ", round(f1_fo, 4),
          " (n models = ", length(fo_matrices), ")")
  frontier_open_f1 <- c(frontier_open_f1, f1_fo)

  # -- local: 4 models, corrected run.
  lo <- align_ensemble(file.path(HERE, "data", DATA_DATE, LOCAL_RUN), qc)
  stopifnot(length(lo$matrices) == 4)
  f1_lo <- unanimous_f1(lo$matrices, lo$truth)
  message("  local unanimous macro_f1 = ", round(f1_lo, 4),
          " (n models = ", length(lo$matrices), ")")
  local_f1 <- c(local_f1, f1_lo)
}

message("")
message("=== SUMMARY (survey-question-level, mean of 3 per-question macro F1s) ===")
message("frontier-open unanimous ensemble: ", round(mean(frontier_open_f1), 4),
        "  (per-q: ", paste(round(frontier_open_f1, 4), collapse = ", "), ")")
message("local unanimous ensemble:         ", round(mean(local_f1), 4),
        "  (per-q: ", paste(round(local_f1, 4), collapse = ", "), ")")
