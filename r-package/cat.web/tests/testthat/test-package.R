test_that("exported functions are callable", {
  expect_true(is.function(classify))
  expect_true(is.function(extract))
  expect_true(is.function(explore))
  expect_true(is.function(summarize))
})
