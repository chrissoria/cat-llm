test_that("exported functions are callable", {
  expect_true(is.function(classify))
  expect_true(is.function(extract))
  expect_true(is.function(explore))
  expect_true(is.function(summarize))
  expect_true(is.function(prompt_tune))
  expect_true(is.function(install_cat_stack))
})

test_that("internal helpers are exported and callable", {
  expect_true(is.function(cat.stack::.strip_quotes))
  expect_true(is.function(cat.stack::.as_py_int))
  expect_true(is.function(cat.stack::.validate_add_other))
  expect_true(is.function(cat.stack::.convert_models))
})

test_that(".strip_quotes removes wrapping quotes", {
  expect_equal(cat.stack::.strip_quotes('"abc"'), "abc")
  expect_equal(cat.stack::.strip_quotes("'abc'"), "abc")
  expect_equal(cat.stack::.strip_quotes("abc"), "abc")
  expect_null(cat.stack::.strip_quotes(NULL))
})

test_that(".validate_add_other accepts valid values, rejects invalid", {
  expect_identical(cat.stack::.validate_add_other(TRUE), TRUE)
  expect_identical(cat.stack::.validate_add_other(FALSE), FALSE)
  expect_identical(cat.stack::.validate_add_other("prompt"), "prompt")
  expect_error(cat.stack::.validate_add_other("invalid"))
  expect_error(cat.stack::.validate_add_other(1))
})

test_that("classify() formals include the new v1.6.0 params", {
  # If a Python-side parameter is exposed in the R wrapper, the user
  # can pass it. These tests don't call the Python side (no API key
  # required) — they just guard against signature drift between
  # ports.
  fm <- names(formals(classify))
  expect_true("batch_mode"           %in% fm)
  expect_true("batch_poll_interval"  %in% fm)
  expect_true("batch_timeout"        %in% fm)
  expect_true("json_retries"         %in% fm)
  expect_true("json_formatter"       %in% fm)
  expect_true("two_step_classify"    %in% fm)
  expect_true("embedding_tiebreaker" %in% fm)
  expect_true("min_centroid_size"    %in% fm)
})

test_that("classify() defaults align with the Python signature", {
  fm <- formals(classify)
  # Tracks Python `batch_retries: int = 1` (lowered in cat-stack 1.4.1)
  expect_equal(eval(fm$batch_retries), 1L)
  # Tracks Python `json_retries: int = 2`
  expect_equal(eval(fm$json_retries), 2L)
  # Tracks Python `embedding_tiebreaker: bool = False`
  expect_false(eval(fm$embedding_tiebreaker))
  # Three-state Optional[bool] = None — R uses NULL for the auto branch
  expect_null(eval(fm$json_formatter))
  expect_null(eval(fm$two_step_classify))
  # Tracks Python `min_centroid_size: int = 3`
  expect_equal(eval(fm$min_centroid_size), 3L)
})

test_that("summarize() batch_retries default matches Python", {
  fm <- formals(summarize)
  expect_equal(eval(fm$batch_retries), 1L)
})
