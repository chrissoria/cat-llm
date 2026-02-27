test_that(".convert_models() handles a single 3-element character vector", {
  skip_if_no_catllm()
  py_list <- catllm:::.convert_models(list(c("gpt-4o", "openai", "sk-test")))
  expect_equal(length(py_list), 1L)
  tup <- py_list[[1]]
  expect_equal(reticulate::py_to_r(tup[[0L]]), "gpt-4o")
  expect_equal(reticulate::py_to_r(tup[[1L]]), "openai")
  expect_equal(reticulate::py_to_r(tup[[2L]]), "sk-test")
})

test_that(".convert_models() handles a 4-element list with options", {
  skip_if_no_catllm()
  entry    <- list("gpt-4o", "openai", "sk-test", list(creativity = 0.5))
  py_list  <- catllm:::.convert_models(list(entry))
  tup      <- py_list[[1]]
  opts     <- reticulate::py_to_r(tup[[3L]])
  expect_equal(opts$creativity, 0.5)
})

test_that(".convert_models() wraps a bare character vector", {
  skip_if_no_catllm()
  py_list <- catllm:::.convert_models(c("gpt-4o", "openai", "sk-test"))
  expect_equal(length(py_list), 1L)
})

test_that(".as_py_int() returns Python None for NULL", {
  skip_if_no_catllm()
  result <- catllm:::.as_py_int(NULL)
  expect_true(reticulate::is_py_none(result))
})

test_that(".as_py_int() converts double to int", {
  skip_if_no_catllm()
  result <- catllm:::.as_py_int(5.9)
  expect_equal(reticulate::py_to_r(result), 5L)
})

test_that(".validate_add_other() accepts valid values", {
  expect_equal(catllm:::.validate_add_other(FALSE),    FALSE)
  expect_equal(catllm:::.validate_add_other(TRUE),     TRUE)
  expect_equal(catllm:::.validate_add_other("prompt"), "prompt")
})

test_that(".validate_add_other() rejects invalid values", {
  expect_error(catllm:::.validate_add_other("yes"), "'add_other' must be")
  expect_error(catllm:::.validate_add_other(1L),    "'add_other' must be")
})

# ── Smoke tests (require OPENAI_API_KEY) ──────────────────────────────────────

test_that("classify() returns a data.frame with correct dimensions", {
  skip_if_no_catllm()
  skip_if_no_api_key("OPENAI_API_KEY")

  texts <- c(
    "I really loved this product!",
    "Absolutely terrible experience.",
    "It was fine, nothing special."
  )
  result <- catllm::classify(
    input_data      = texts,
    categories      = c("Positive", "Negative", "Neutral"),
    description     = "Customer feedback",
    api_key         = Sys.getenv("OPENAI_API_KEY"),
    add_other       = FALSE,
    check_verbosity = FALSE
  )

  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 3L)
  expect_true(any(c("Positive", "Negative", "Neutral") %in% names(result)))
})

# ── Ensemble tests (require both OPENAI_API_KEY and ANTHROPIC_API_KEY) ────────

test_that("classify() ensemble returns consensus columns", {
  skip_if_no_catllm()
  skip_if_no_api_key("OPENAI_API_KEY")
  skip_if_no_api_key("ANTHROPIC_API_KEY")

  texts <- c("I love this!", "Terrible.", "It was okay.")

  result <- catllm::classify(
    input_data  = texts,
    categories  = c("Positive", "Negative", "Neutral"),
    models      = list(
      c("gpt-4o",              "openai",    Sys.getenv("OPENAI_API_KEY")),
      c("claude-haiku-4-5-20251001", "anthropic", Sys.getenv("ANTHROPIC_API_KEY"))
    ),
    consensus_threshold = "unanimous",
    add_other           = FALSE,
    check_verbosity     = FALSE
  )

  expect_s3_class(result, "data.frame")
  expect_equal(nrow(result), 3L)
  # Ensemble output should include at least one consensus column
  expect_true(any(grepl("consensus", names(result), ignore.case = TRUE)))
})
