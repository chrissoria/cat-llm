test_that("extract() returns a named list with correct structure", {
  skip_if_no_catllm()
  skip_if_no_api_key("OPENAI_API_KEY")

  texts <- c(
    "I moved for a new job opportunity.",
    "We relocated to be closer to family.",
    "The cost of living was too high.",
    "I wanted a change of scenery.",
    "My company transferred me.",
    "Better schools for the kids.",
    "I moved for love.",
    "Lower taxes in the new state."
  )

  result <- catllm::extract(
    input_data  = texts,
    description = "Why did you move?",
    api_key     = Sys.getenv("OPENAI_API_KEY"),
    iterations  = 1L,
    divisions   = 2L,
    max_categories = 5L
  )

  # Should return a list
  expect_type(result, "list")

  # Must contain top_categories as a character vector
  expect_true("top_categories" %in% names(result))
  expect_type(result$top_categories, "character")
  expect_gte(length(result$top_categories), 1L)

  # Must contain counts_df as a data.frame
  expect_true("counts_df" %in% names(result))
  expect_s3_class(result$counts_df, "data.frame")
})
