# Changelog

All notable changes to CatLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [2.0.0] - 2025-01-17

### Major Release: Simplified API & Ensemble Methods

Version 2.0 represents a major simplification of CatLLM's architecture and API, making it easier to install, use, and extend.

### Added
- **Ensemble classification**: Run multiple models in parallel and combine predictions
  - Cross-provider ensembles (GPT-4o + Claude + Gemini)
  - Self-consistency ensembles (same model with temperature variation)
  - Model comparison mode for side-by-side evaluation
- **Consensus voting methods**:
  - `"majority"` - 50%+ agreement required
  - `"two-thirds"` - 67%+ agreement required
  - `"unanimous"` - 100% agreement required
  - Custom numeric thresholds (e.g., `0.75` for 75%)
- **Visualization tools** in web app:
  - Classification matrix heatmap
  - Category distribution charts
  - Download buttons for all visualizations
- PDF report generation with methodology documentation

### Changed
- **Simplified to 3 core functions**:
  - `extract()` - Discover categories in your data
  - `classify()` - Assign categories to your data
  - `summarize()` - Generate summaries of your data
- **Removed SDK dependencies**: All API calls now use pure `requests` library
  - No more `openai`, `anthropic`, `google-generativeai` package requirements
  - Lighter installation, fewer dependency conflicts
  - Unified HTTP interface for all providers
- **Streamlined parameters**: Consistent parameter names across all functions
- Web app UI improvements: button alignment, Garamond font, improved layout

### Removed
- Direct SDK dependencies (openai, anthropic, google-generativeai, mistralai)
- Legacy function names (old aliases still work but are deprecated)

### Migration from 1.x
Most code will work without changes. Key differences:
- SDK-specific features (like streaming) are no longer available
- All providers now use the same HTTP-based interface
- New `models` parameter enables ensemble mode

---

## [0.1.15] - 2025-01-10

### Added
- `summarize()` function for text and PDF summarization with multi-model support
- `focus` parameter for `extract()` to prioritize specific themes during category discovery
- `progress_callback` parameter for PDF page-by-page progress updates
- Multi-model support in `classify()` via `models` parameter for ensemble classification
- Documentation for `summarize()` function in README

### Changed
- Converted web app from Gradio to Streamlit for better mobile support
- Improved PDF functionality in HuggingFace app

### Fixed
- Parameter mapping in `classify()` function
- Bug in extract function for edge cases
- Extract API now uses chat.completions for OpenAI-compatible providers

---

## [0.1.14] - 2025-01-02

### Added
- **Ollama support** for local model inference (llama3, mistral, etc.)
- Auto-download of Ollama models when not installed
- System resource checks before downloading large models
- Confirmation prompts before downloading Ollama models

### Changed
- Improved error messages and download warnings for Ollama integration

---

## [0.1.13] - 2024-12-30

### Added
- Unified HTTP-based multi-class text classification
- Multiple categories per item for PDFs and images
- Extract categories functionality for PDFs and images

### Changed
- Web app made mobile-friendly
- Auto-adjust `divisions` and `categories_per_chunk` for small datasets
- Aligned PDF function output format with text classifier

### Fixed
- Image classification output alignment with other classifiers
- Glitch causing errors in app when using image classification

---

## [0.1.12] - 2024-12-15

### Added
- **PDF document classification** with multiple processing modes:
  - `image` mode: renders pages as images for visual analysis
  - `text` mode: extracts text for text-based classification
  - `both` mode: combines image and text analysis
- **HuggingFace Spaces web app** for browser-based classification

### Changed
- Moved web app to CatLLM organization on HuggingFace

---

## [0.1.11] - 2024-12-01

### Added
- **Image classification** using vision models
- Image file upload support with description context
- Support for multiple image formats (PNG, JPG, JPEG, GIF, WEBP)

---

## [0.1.10] - 2024-11-20

### Added
- **Chain of Verification (CoVe)** prompting for improved accuracy
- **Step-back prompting** option for complex classifications
- **Context prompting** to add expert domain knowledge
- Warning messages for CoVe users about processing time

### Changed
- Refactored and tested multi_class function
- Cleaned up prompt code structure

### Fixed
- CoT prompt not producing structured output in some cases
- Error handling improvements for Google, OpenAI, and Mistral providers

---

## [0.1.9] - 2024-11-15

### Added
- **HuggingFace Inference API** support as model provider
- Auto-detection of model source based on model name
- Few-shot learning with `example1` through `example6` parameters

### Changed
- Default model for text classification set to GPT-4o

---

## [0.1.8] - 2024-11-10

### Added
- **Perplexity** as web search provider
- Advanced search with dates and confidence scores
- Formal URL output in web search function

### Changed
- Web search method no longer halts on rate limit
- Removed case sensitivity for `model_source` input

---

## [0.1.7] - 2024-11-05

### Added
- **Google search** capabilities for web search function
- Web search dataset building function
- Example script for categorizing text data

### Changed
- `creativity` parameter now optional (uses model defaults)
- Improved column names for easier understanding

### Fixed
- Error message when model is not valid
- Image inputs with file paths no longer crash the function

---

## [0.1.6] - 2024-10-25

### Added
- **xAI (Grok)** support for text classification
- Auto-create categories option in multi_class function
- Rate limit handling for OpenAI and Google

### Fixed
- Issue where whole row was converted to missing if one category wasn't output
- HuggingFace retry when incorrect JSON format is returned
- Column converting to 0s for valid rows
- Explore corpus failure when non-string value in rows

---

## [0.1.5] - 2024-10-15

### Added
- **Google (Gemini)** support for multi-class text classification
- **Anthropic (Claude)** support for CERAD and image functions
- **Mistral** support for CERAD and image functions
- Reference images provided within package for CERAD scoring

### Changed
- Updated license to be JOSS-acceptable (MIT)

---

## [0.1.4] - 2024-10-01

### Added
- `explore_common_categories()` function for automatic category discovery
- Research question parameter for guided category extraction
- Specificity parameter ("broad" or "specific") for category granularity

---

## [0.1.3] - 2024-09-15

### Added
- **CERAD cognitive assessment** scoring functions
- Support for reference images in CERAD analysis
- Option to specify whether image contains a reference

### Changed
- Separated CERAD functions into dedicated module

---

## [0.1.2] - 2024-09-01

### Added
- Image classification functions with OpenAI vision models
- UCNets example usage documentation

### Changed
- Package can now be imported as `catllm` instead of `cat_llm`

---

## [0.1.1] - 2024-08-15

### Added
- Logo and branding
- Improved README documentation

### Fixed
- Various small fixes and improvements

---

## [0.1.0] - 2024-08-01

### Added
- **Initial release**
- `classify()` function for multi-class text classification
- Support for OpenAI models (GPT-4, GPT-4o, GPT-3.5)
- Binary classification output (0/1) for each category
- CSV export functionality
- Basic error handling and retry logic

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| **2.0.0** | **2025-01-17** | **Simplified API, ensemble methods, removed SDK dependencies** |
| 0.1.15 | 2025-01-10 | Summarization, focus parameter, Streamlit web app |
| 0.1.14 | 2025-01-02 | Ollama local inference |
| 0.1.13 | 2024-12-30 | Multi-category support, mobile web app |
| 0.1.12 | 2024-12-15 | PDF classification, HuggingFace app |
| 0.1.11 | 2024-12-01 | Image classification |
| 0.1.10 | 2024-11-20 | CoVe, step-back, context prompting |
| 0.1.9 | 2024-11-15 | HuggingFace support, few-shot learning |
| 0.1.8 | 2024-11-10 | Perplexity web search |
| 0.1.7 | 2024-11-05 | Google search, web search datasets |
| 0.1.6 | 2024-10-25 | xAI/Grok support, auto-categories |
| 0.1.5 | 2024-10-15 | Google/Anthropic/Mistral providers |
| 0.1.4 | 2024-10-01 | Category discovery function |
| 0.1.3 | 2024-09-15 | CERAD cognitive scoring |
| 0.1.2 | 2024-09-01 | Image classification |
| 0.1.1 | 2024-08-15 | Branding, documentation |
| 0.1.0 | 2024-08-01 | Initial release |

---

[Unreleased]: https://github.com/chrissoria/cat-llm/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/chrissoria/cat-llm/compare/v0.1.15...v2.0.0
[0.1.15]: https://github.com/chrissoria/cat-llm/compare/v0.1.14...v0.1.15
[0.1.14]: https://github.com/chrissoria/cat-llm/compare/v0.1.13...v0.1.14
[0.1.13]: https://github.com/chrissoria/cat-llm/compare/v0.1.12...v0.1.13
[0.1.12]: https://github.com/chrissoria/cat-llm/compare/v0.1.11...v0.1.12
[0.1.11]: https://github.com/chrissoria/cat-llm/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/chrissoria/cat-llm/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/chrissoria/cat-llm/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/chrissoria/cat-llm/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/chrissoria/cat-llm/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/chrissoria/cat-llm/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/chrissoria/cat-llm/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/chrissoria/cat-llm/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/chrissoria/cat-llm/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/chrissoria/cat-llm/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/chrissoria/cat-llm/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/chrissoria/cat-llm/releases/tag/v0.1.0
