# Changelog

All notable changes to CatLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Ensemble classification: run multiple models in parallel with consensus voting
- String values for `consensus_threshold` parameter ("majority", "two-thirds", "unanimous")
- Classification matrix heatmap visualization in web app
- Category distribution charts in web app

### Changed
- Web app: moved "Try Example Dataset" button next to file uploader
- Web app: improved radio button styling to fill available width
- Web app: applied Garamond font globally

## [0.1.15] - 2025-01-10

### Added
- `summarize()` function for text and PDF summarization
- `focus` parameter for category extraction to prioritize specific themes
- `progress_callback` parameter for PDF page-by-page progress updates
- Multi-model support in `classify()` with `models` parameter

### Fixed
- Parameter mapping in `classify()` function
- PDF processing mode handling

## [0.1.14] - 2025-01-02

### Added
- Ollama support for local model inference
- Auto-download of Ollama models when not installed
- System resource checks before downloading large models

### Changed
- Improved error messages for Ollama integration
- Extract function now uses HTTP-based chat completions for OpenAI-compatible providers

### Fixed
- Bug in extract function for certain edge cases

## [0.1.13] - 2024-12-30

### Added
- Unified HTTP-based multi-class text classification
- Mobile-friendly web app interface
- Multiple categories per item for PDFs and images

### Changed
- Auto-adjust `divisions` and `categories_per_chunk` for small datasets
- Aligned PDF function output format with text classifier

## [0.1.12] - 2024-12-15

### Added
- Extract categories functionality for PDFs and images
- Web app hosted on HuggingFace Spaces

### Fixed
- Image classification output alignment with other classifiers

## [0.1.11] - 2024-12-01

### Added
- PDF document classification with multiple processing modes (image, text, both)
- Image classification using vision models
- Support for multiple LLM providers (OpenAI, Anthropic, Google, Mistral, Groq, xAI)

## [0.1.10] - 2024-11-15

### Added
- Initial public release
- `extract()` function for discovering categories in text data
- `classify()` function for assigning predefined categories
- Chain of thought reasoning option
- Few-shot learning with example parameters

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.15 | 2025-01-10 | Summarization, focus parameter, multi-model support |
| 0.1.14 | 2025-01-02 | Ollama local inference |
| 0.1.13 | 2024-12-30 | Mobile web app, multi-category support |
| 0.1.12 | 2024-12-15 | PDF/image extraction, HuggingFace app |
| 0.1.11 | 2024-12-01 | PDF/image classification, multi-provider |
| 0.1.10 | 2024-11-15 | Initial release |

[Unreleased]: https://github.com/chrissoria/cat-llm/compare/v0.1.15...HEAD
[0.1.15]: https://github.com/chrissoria/cat-llm/compare/v0.1.14...v0.1.15
[0.1.14]: https://github.com/chrissoria/cat-llm/compare/v0.1.13...v0.1.14
[0.1.13]: https://github.com/chrissoria/cat-llm/compare/v0.1.12...v0.1.13
[0.1.12]: https://github.com/chrissoria/cat-llm/compare/v0.1.11...v0.1.12
[0.1.11]: https://github.com/chrissoria/cat-llm/compare/v0.1.10...v0.1.11
[0.1.10]: https://github.com/chrissoria/cat-llm/releases/tag/v0.1.10
