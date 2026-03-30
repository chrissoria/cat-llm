#' Domain-suffixed aliases for the CatLLM ecosystem
#'
#' These functions provide convenient domain-suffixed names so users can
#' tab-complete to find the right function. Each is a thin re-export from
#' the corresponding domain package.
#'
#' @name catllm-aliases
NULL

# --- Survey (from cat.survey) ---

#' @rdname catllm-aliases
#' @inheritParams cat.survey::classify
#' @export
classify_survey <- function(...) cat.survey::classify(...)

#' @rdname catllm-aliases
#' @inheritParams cat.survey::extract
#' @export
extract_survey <- function(...) cat.survey::extract(...)

#' @rdname catllm-aliases
#' @inheritParams cat.survey::explore
#' @export
explore_survey <- function(...) cat.survey::explore(...)

# --- Social Media (from cat.vader) ---

#' @rdname catllm-aliases
#' @inheritParams cat.vader::classify
#' @export
classify_social <- function(...) cat.vader::classify(...)

#' @rdname catllm-aliases
#' @inheritParams cat.vader::extract
#' @export
extract_social <- function(...) cat.vader::extract(...)

#' @rdname catllm-aliases
#' @inheritParams cat.vader::explore
#' @export
explore_social <- function(...) cat.vader::explore(...)

# --- Academic (from cat.ademic) ---

#' @rdname catllm-aliases
#' @inheritParams cat.ademic::classify
#' @export
classify_academic <- function(...) cat.ademic::classify(...)

#' @rdname catllm-aliases
#' @inheritParams cat.ademic::extract
#' @export
extract_academic <- function(...) cat.ademic::extract(...)

#' @rdname catllm-aliases
#' @inheritParams cat.ademic::explore
#' @export
explore_academic <- function(...) cat.ademic::explore(...)

#' @rdname catllm-aliases
#' @inheritParams cat.ademic::summarize
#' @export
summarize_academic <- function(...) cat.ademic::summarize(...)

# --- Cognitive (from cat.cog) ---

#' @rdname catllm-aliases
#' @inheritParams cat.cog::cerad_drawn_score
#' @export
cerad_drawn_score <- function(...) cat.cog::cerad_drawn_score(...)
