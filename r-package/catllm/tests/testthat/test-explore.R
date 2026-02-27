test_that("explore() returns a character vector", {
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

  result <- catllm::explore(
    input_data  = texts,
    description = "Why did you move?",
    api_key     = Sys.getenv("OPENAI_API_KEY"),
    iterations  = 1L,
    divisions   = 2L
  )

  expect_type(result, "character")
  expect_gte(length(result), 1L)
})

test_that("explore() length is approximately iterations * divisions * categories_per_chunk", {
  skip_if_no_catllm()
  skip_if_no_api_key("OPENAI_API_KEY")

  texts <- c(
    "I moved for work.",
    "Family reasons.",
    "Better schools.",
    "Cheaper housing.",
    "New adventure.",
    "Retirement.",
    "Climate.",
    "Partner's job."
  )

  iterations        <- 2L
  divisions         <- 3L
  categories_per_chunk <- 5L

  result <- catllm::explore(
    input_data           = texts,
    description          = "Why did you move?",
    api_key              = Sys.getenv("OPENAI_API_KEY"),
    iterations           = iterations,
    divisions            = divisions,
    categories_per_chunk = categories_per_chunk
  )

  # Allow 50% tolerance — models may return fewer/more items per chunk
  expected <- iterations * divisions * categories_per_chunk
  expect_gte(length(result), expected * 0.5)
  expect_lte(length(result), expected * 2.0)
})
