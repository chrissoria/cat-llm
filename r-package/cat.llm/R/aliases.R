#' Domain-suffixed aliases for the CatLLM ecosystem
#'
#' These functions provide convenient domain-suffixed names so users can
#' tab-complete to find the right function. Each is a thin re-export from
#' the corresponding domain package.
#'
#' @name catllm-aliases
#' @examples
#' \dontrun{
#' library(cat.llm)
#'
#' # Survey classification (re-export of cat.survey::classify)
#' classify_survey(
#'   input_data      = df$responses,
#'   categories      = c("Cost", "Quality", "Service", "Other"),
#'   survey_question = "Why did you choose us?",
#'   api_key         = Sys.getenv("OPENAI_API_KEY")
#' )
#'
#' # Political documents (re-export of cat.pol::classify)
#' classify_political(
#'   source     = "city_san_diego",
#'   doc_type   = "ordinance",
#'   n          = 50L,
#'   categories = c("Housing", "Public Safety", "Finance"),
#'   api_key    = Sys.getenv("OPENAI_API_KEY")
#' )
#'
#' # Web content (re-export of cat.web::classify)
#' classify_web(
#'   input_data    = c("https://example.com/article-1",
#'                     "https://example.com/article-2"),
#'   categories    = c("News", "Opinion", "Tutorial"),
#'   source_domain = "example.com",
#'   api_key       = Sys.getenv("OPENAI_API_KEY")
#' )
#'
#' # CERAD cognitive scoring (re-export of cat.cog::cerad_drawn_score)
#' cerad_drawn_score(
#'   shape       = "circle",
#'   image_input = "./drawings/",
#'   api_key     = Sys.getenv("OPENAI_API_KEY")
#' )
#' }
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

# --- Political / Policy (from cat.pol) ---

#' @rdname catllm-aliases
#' @inheritParams cat.pol::classify
#' @export
classify_political <- function(...) cat.pol::classify(...)

#' @rdname catllm-aliases
#' @inheritParams cat.pol::extract
#' @export
extract_political <- function(...) cat.pol::extract(...)

#' @rdname catllm-aliases
#' @inheritParams cat.pol::explore
#' @export
explore_political <- function(...) cat.pol::explore(...)

#' @rdname catllm-aliases
#' @inheritParams cat.pol::summarize
#' @export
summarize_political <- function(...) cat.pol::summarize(...)

# --- Web (from cat.web) ---

#' @rdname catllm-aliases
#' @inheritParams cat.web::classify
#' @export
classify_web <- function(...) cat.web::classify(...)

#' @rdname catllm-aliases
#' @inheritParams cat.web::extract
#' @export
extract_web <- function(...) cat.web::extract(...)

#' @rdname catllm-aliases
#' @inheritParams cat.web::explore
#' @export
explore_web <- function(...) cat.web::explore(...)

#' @rdname catllm-aliases
#' @inheritParams cat.web::summarize
#' @export
summarize_web <- function(...) cat.web::summarize(...)
