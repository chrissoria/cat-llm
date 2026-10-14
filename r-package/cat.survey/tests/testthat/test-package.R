test_that("exported functions are callable", {
  expect_true(is.function(classify))
  expect_true(is.function(extract))
  expect_true(is.function(explore))
})

test_that("classify() exposes the batch and system-prompt arguments", {
  args <- names(formals(classify))
  expect_true(all(c("batch_mode", "batch_poll_interval", "batch_timeout",
                    "system_prompt") %in% args))
  expect_false(formals(classify)$batch_mode)
  expect_identical(formals(classify)$system_prompt, "")
  expect_true("json_formatter" %in% args)
  expect_null(formals(classify)$json_formatter)
})
