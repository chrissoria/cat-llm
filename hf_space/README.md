---
title: catllm Classifier
emoji: 🏷️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.6.0"
app_file: app.py
pinned: false
license: mit
short_description: Classify text, images, and PDFs using LLMs
---

# catllm Classifier

A web interface for the [catllm](https://github.com/chrissoria/cat-llm) Python package. Classify text, images, and PDF documents into custom categories using various LLM providers.

## How to Use

1. **Select Input Type**: Choose between Text, Image, or PDF
2. **Provide Your Data**: Enter text directly, or upload an image/PDF file
3. **Define Categories**: Enter comma-separated categories (e.g., "Positive, Negative, Neutral")
4. **Add Context** (optional): Describe what you're classifying to help the model
5. **Choose a Model**: Select your preferred LLM
6. **Enter API Key**: Provide your API key for the selected model's provider
7. **Click Classify**: View results showing which categories apply

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
