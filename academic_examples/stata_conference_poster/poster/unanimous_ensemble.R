# =============================================================================
# unanimous_ensemble.R — numbers behind the poster's "Unanimous Ensembles"
# chart (fig4_unanimous.png). The chart plots pooled precision and recall
# (all response x category cells flattened into one confusion matrix, the
# same cell-level pooling as llm_ensemble_paper's unanimous_vote() /
# compute_metrics() pipeline) for three groups: a top-3 unanimous ensemble
# of frontier-open models, a top-3 unanimous ensemble of local models, and
# Claude Fable 5 as a single model.
#
# "Top-3" = the three models in the tier with the highest individual macro
# F1 (mean over the three survey questions), a fixed trio used for every
# question. Chosen because it is the rule a reader can actually apply
# ("run your three best models, require unanimity"), and because it is
# what llm_ensemble_paper does when it adds models best-first. The trio is
# ranked on the same 175 rows per question it is scored on, so this is
# selection on the scored data; as a robustness check the script also
# scores every 3-model subset of the tier and reports where the top-3 trio
# lands and how many subsets beat Claude Fable 5. The full-tier unanimous
# ensembles (all 6 / all 4 models) are computed and printed for reference.
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
# Last run 2026-09-22 (each headline number is the mean of the three
# per-question values; per-question order a19i, a19f, e1b):
#   frontier-open top-3 unanimous (Kimi K2.6, Qwen3.5 397B, Gemma 4 31B):
#     precision 0.8509 (0.8786, 0.9450, 0.7290)   <- plotted
#     recall    0.8450 (0.8849, 0.9115, 0.7386)   <- plotted
#     pooled F1 0.8478 (0.8817, 0.9279, 0.7338)
#     macro F1  0.8345 (0.8934, 0.9404, 0.6696)
#     robustness: 2nd of 20 three-model subsets on pooled F1 (best 0.8492,
#     a tie); all 20 subsets beat Fable 5 on precision, 17 of 20 on pooled F1.
#   frontier-open all-6 unanimous (reference):
#     precision 0.8691, recall 0.7967, pooled F1 0.8311, macro F1 0.8044
#   local top-3 unanimous (Qwen3 14B, Gemma3 12B, Llama 3.1 8B):
#     precision 0.8173 (0.9138, 0.8627, 0.6754)   <- plotted
#     recall    0.6815 (0.7626, 0.7788, 0.5033)   <- plotted
#     pooled F1 0.7423 (0.8314, 0.8186, 0.5768)
#     macro F1  0.7177 (0.8167, 0.8359, 0.5005)
#     robustness: 1st of 4 three-model subsets on pooled F1; all 4 beat
#     Fable 5 on precision, none on pooled F1.
#   local all-4 unanimous (reference):
#     precision 0.8622, recall 0.5858, pooled F1 0.6949, macro F1 0.6693
#   Claude Fable 5 (single model):
#     precision 0.7827 (0.8832, 0.8487, 0.6161)   <- plotted
#     recall    0.8713 (0.8705, 0.8938, 0.8497)   <- plotted
#     pooled F1 0.8206, macro F1 0.8132
#   All 175/175 gold-standard rows matched exactly for every question/tier
#   (no fuzzy fallback needed); Fable 5's own reference-run rows (423-1090
#   per question, its full gold pool) matched 100% exact too.
# =============================================================================

HERE <- Sys.getenv("CATLLM_ENSEMBLE_DATA_ROOT", unset = "")
if (!nzchar(HERE)) {
  stop("Set CATLLM_ENSEMBLE_DATA_ROOT to the companion benchmark's local ",
       "path before running (see header comment).")
}
# Expected layout under HERE:
#   R/scoring_lib.R                       shared gold-standard alignment/scoring
#   data/<date>/<run>/{a19i,a19f,e1b}_ensemble.csv   per-model 0/1 predictions
#   data/fable5_reference_<date>/{a19i,a19f,e1b}_fable5_full*.csv
#                                          Claude Fable 5's own predictions
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
FABLE5_DIR <- file.path(HERE, "data", "fable5_reference_07_04_26")
TOP_N <- 3

# Score a prediction matrix from one compute_metrics() call:
#   macro  -- per-category F1 (NA if no gold support), mean over supported
#             categories. What fig2/fig3 use, and what ranks members.
#   pooled -- every response x category cell pooled into one confusion
#             matrix before computing F1 (tp/fp/fn already summed across
#             categories in compute_metrics()'s "overall" row). What
#             llm_ensemble_paper's unanimous_vote() does.
#   prec / recall -- pooled precision and sensitivity from the same cells.
#             These are what fig4 plots: the over-classification argument is
#             a precision claim, and unanimity pays for it in recall, so the
#             chart shows both sides of that trade directly.
score_all <- function(pred, truth, label) {
  stats <- compute_metrics(pred, truth, label = label)
  ov <- stats[stats$category == "overall", ]
  c(macro = ov$macro_f1,
    pooled = 2 * ov$tp / (2 * ov$tp + ov$fp + ov$fn),
    prec = ov$tp / (ov$tp + ov$fp),
    recall = ov$tp / (ov$tp + ov$fn))
}

unanimous_pred_of <- function(matrices_list, truth) {
  n_models <- length(matrices_list)
  stacked <- array(unlist(matrices_list), dim = c(nrow(truth), ncol(truth), n_models))
  apply(stacked, c(1, 2), function(v) as.integer(all(v == 1)))
}

# ---------------------------------------------------------------- load
# One pass to align everything; all scoring happens on the loaded matrices.
D <- list()
for (qc in QUESTIONS) {
  message("=== loading ", qc, " ===")

  # -- frontier-open: the 5-model run plus the 6th model's own canonical run.
  # Both are independently aligned to the same gold standard via
  # align_ensemble(); the 6th model's matrix is then reindexed onto the main
  # run's gold-row order (not raw row position, which is not guaranteed to
  # match across files) before any combine.
  main <- align_ensemble(file.path(HERE, "data", DATA_DATE, FRONTIER_OPEN_MAIN_RUN), qc)
  extra_run <- align_ensemble(file.path(HERE, "data", DATA_DATE, FRONTIER_OPEN_EXTRA_RUN), qc)
  stopifnot(length(main$matrices) == 5)
  stopifnot(FRONTIER_OPEN_EXTRA_MODEL %in% names(extra_run$matrices))
  extra_by_goldidx <- extra_run$matrices[[FRONTIER_OPEN_EXTRA_MODEL]]
  rownames(extra_by_goldidx) <- extra_run$idx
  extra_aligned <- extra_by_goldidx[as.character(main$idx), ]
  stopifnot(!any(is.na(extra_aligned)))
  fo <- c(main$matrices, setNames(list(extra_aligned), FRONTIER_OPEN_EXTRA_MODEL))
  stopifnot(length(fo) == 6)

  # -- local: 4 models, corrected run.
  lo <- align_ensemble(file.path(HERE, "data", DATA_DATE, LOCAL_RUN), qc)
  stopifnot(length(lo$matrices) == 4)

  # -- Claude Fable 5: single model, own reference run (subscription agent
  # access, thinking on -- see fable5_reference.csv's own build script for
  # why it's a reference point rather than an 18th leaderboard model).
  # Matched to the SAME 175-row subsample as the tier ensembles above (the
  # full gold pool the reference run was scored against has to be narrowed
  # to that subsample for the numbers to be comparable at all), same
  # gold-index re-keying pattern as the frontier-open combine above.
  gold <- load_gold(qc)
  f5_path <- Sys.glob(file.path(FABLE5_DIR, sprintf("%s_fable5_full*.csv", qc)))
  stopifnot(length(f5_path) == 1)
  f5 <- read.csv(f5_path, stringsAsFactors = FALSE)
  stopifnot(all(f5$processing_status == "success"))
  fmat <- as.matrix(f5[, sprintf("category_%d", 1:6)])
  storage.mode(fmat) <- "integer"
  f_idx <- match_gold(f5$input_data, gold, qc = qc)
  map <- rep(NA_integer_, nrow(gold))
  for (g in unique(f_idx)) {
    rows <- which(f_idx == g)
    if (length(rows) > 1) {
      sub <- fmat[rows, , drop = FALSE]
      if (!all(apply(sub, 2, function(col) length(unique(col)) == 1)))
        stop(sprintf("[%s] duplicate-key Fable 5 rows disagree at gold row %d", qc, g))
    }
    map[g] <- rows[1]
  }
  stopifnot(!any(is.na(map[main$idx])))

  D[[qc]] <- list(fo = fo, fo_truth = main$truth,
                  lo = lo$matrices, lo_truth = lo$truth,
                  fable = fmat[map[main$idx], , drop = FALSE])
}

# ---------------------------------------------------------------- score
# f(qc) -> named metric vector; returns per-question matrix + 3-question mean.
over_q <- function(f) {
  m <- sapply(QUESTIONS, f)
  list(per_q = m, mean = rowMeans(m))
}
fmt <- function(name, r) {
  sprintf("  %-9s %.4f  (%s)", name, r$mean[[name]],
          paste(sprintf("%.4f", r$per_q[name, ]), collapse = ", "))
}
show <- function(title, r) {
  message(title)
  for (n in c("prec", "recall", "pooled", "macro")) message(fmt(n, r))
}

fable <- over_q(function(qc) score_all(D[[qc]]$fable, D[[qc]]$fo_truth, "fable"))

run_tier <- function(tier, key, tkey) {
  models <- names(D[[1]][[key]])
  message("\n================ ", tier, " (", length(models), " models) ================")

  # individual members, ranked by mean macro F1
  indiv <- t(sapply(models, function(m)
    over_q(function(qc) score_all(D[[qc]][[key]][[m]], D[[qc]][[tkey]], m))$mean))
  indiv <- indiv[order(-indiv[, "macro"]), ]
  message("individual members (3-question mean), ranked by macro F1:")
  for (m in rownames(indiv))
    message(sprintf("  %-26s macro %.4f  pooled %.4f  prec %.4f  recall %.4f",
                    m, indiv[m, "macro"], indiv[m, "pooled"], indiv[m, "prec"], indiv[m, "recall"]))
  top <- rownames(indiv)[seq_len(TOP_N)]

  una_of <- function(cm) over_q(function(qc)
    score_all(unanimous_pred_of(D[[qc]][[key]][cm], D[[qc]][[tkey]]), D[[qc]][[tkey]],
              paste(qc, tier, "unanimous", sep = "__")))

  top_r <- una_of(top)
  show(sprintf("\nTOP-%d UNANIMOUS (%s):", TOP_N, paste(top, collapse = ", ")), top_r)
  full_r <- una_of(models)
  show(sprintf("\nALL-%d UNANIMOUS (reference):", length(models)), full_r)

  # robustness: every TOP_N-subset of the tier
  combos <- combn(models, TOP_N, simplify = FALSE)
  sub <- t(sapply(combos, function(cm) una_of(cm)$mean))
  key_of <- function(cm) paste(sort(cm), collapse = "+")
  rownames(sub) <- sapply(combos, key_of)
  rank_top <- rank(-sub[, "pooled"])[key_of(top)]
  message(sprintf("\nrobustness over all %d %d-model subsets:", nrow(sub), TOP_N))
  message(sprintf("  top-%d trio ranks %d of %d on pooled F1 (best subset %.4f)",
                  TOP_N, rank_top, nrow(sub), max(sub[, "pooled"])))
  message(sprintf("  subsets beating Fable 5 on precision: %d of %d (min subset prec %.4f vs Fable %.4f)",
                  sum(sub[, "prec"] > fable$mean[["prec"]]), nrow(sub), min(sub[, "prec"]), fable$mean[["prec"]]))
  message(sprintf("  subsets beating Fable 5 on pooled F1:  %d of %d",
                  sum(sub[, "pooled"] > fable$mean[["pooled"]]), nrow(sub)))
  invisible(list(top = top, top_r = top_r, full_r = full_r))
}

fo_res <- run_tier("frontier_open", "fo", "fo_truth")
lo_res <- run_tier("local", "lo", "lo_truth")
show("\n================ CLAUDE FABLE 5 (single model) ================", fable)

message("\n=== PLOTTED (fig4): precision / recall, 3-question mean ===")
message(sprintf("frontier-open top-%d: %.4f / %.4f", TOP_N, fo_res$top_r$mean[["prec"]], fo_res$top_r$mean[["recall"]]))
message(sprintf("local top-%d:         %.4f / %.4f", TOP_N, lo_res$top_r$mean[["prec"]], lo_res$top_r$mean[["recall"]]))
message(sprintf("Claude Fable 5:      %.4f / %.4f", fable$mean[["prec"]], fable$mean[["recall"]]))
