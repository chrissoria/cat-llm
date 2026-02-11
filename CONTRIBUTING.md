# Contributing to CatLLM

Thank you for your interest in contributing to CatLLM! This document provides guidelines for contributing to the project.

## Reporting Issues

If you encounter a bug, unexpected behavior, or have a feature request, please [open an issue on GitHub](https://github.com/chrissoria/cat-llm/issues). When reporting a bug, include:

- A clear description of the problem
- Steps to reproduce the issue
- Your Python version and `cat-llm` version (`pip show cat-llm`)
- The provider and model you were using (e.g., OpenAI GPT-4o)
- The full error traceback, if applicable

## Seeking Support

- **Usage questions**: Open a [GitHub Discussion](https://github.com/chrissoria/cat-llm/discussions) or [Issue](https://github.com/chrissoria/cat-llm/issues) describing what you're trying to do
- **Documentation gaps**: If something is unclear in the README or examples, open an issue so we can improve it
- **Research collaboration**: For academic collaboration inquiries, email [ChrisSoria@Berkeley.edu](mailto:ChrisSoria@Berkeley.edu)

## Contributing Code

### Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/cat-llm.git
   cd cat-llm
   ```
3. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b your-feature-name
   ```

### Making Changes

- Keep changes focused — one feature or fix per pull request
- Follow the existing code style and conventions in the project
- Add or update tests if applicable
- Update documentation (README, docstrings) if your change affects the public API

### Submitting a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin your-feature-name
   ```
2. Open a pull request against the `main` branch of [chrissoria/cat-llm](https://github.com/chrissoria/cat-llm)
3. Describe what your PR does and why, linking any related issues
4. A maintainer will review your PR and may request changes

### Areas Where Contributions Are Welcome

- Bug fixes and error handling improvements
- Support for additional LLM providers
- Local model testing and integration (e.g., Ollama, llama.cpp)
- Support for specialized models (e.g., domain-specific or fine-tuned models)
- Documentation improvements and usage examples
- Performance optimizations
- Test coverage

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive environment for everyone.
