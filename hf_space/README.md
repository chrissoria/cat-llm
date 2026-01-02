---
title: catllm - Survey Response Classifier
emoji: 🏷️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.6.0"
app_file: app.py
pinned: false
license: mit
short_description: Classify survey responses using LLMs
---

# catllm - Survey Response Classifier

A web interface for the [catllm](https://github.com/chrissoria/cat-llm) Python package. Classify survey responses into custom categories using various LLM providers.

## How to Use

1. **Upload Your Data**: Upload a CSV or Excel file containing survey responses
2. **Select Column**: Choose the column containing the text responses to classify
3. **Define Categories**: Enter your classification categories (e.g., "Positive", "Negative", "Neutral")
4. **Choose a Model**: Select your preferred LLM (free models available!)
5. **Click Classify**: View and download results with category assignments

## Supported Models

| Provider | Models |
|----------|--------|
| **OpenAI** | gpt-4o, gpt-4o-mini |
| **Anthropic** | claude-3-5-sonnet, claude-3-haiku |
| **Google** | gemini-1.5-pro, gemini-1.5-flash |
| **Mistral** | mistral-large-latest |

## Privacy

Your API key is **never stored**. It is only used for the current classification request and is not logged or saved.

## Learn More

- [catllm on PyPI](https://pypi.org/project/cat-llm/)
- [GitHub Repository](https://github.com/chrissoria/cat-llm)
- [Documentation](https://github.com/chrissoria/cat-llm#readme)
