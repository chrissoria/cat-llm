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
