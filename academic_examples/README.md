<table>
    <tr>
       <td>
          <img src="https://raw.githubusercontent.com/chrissoria/cat-llm/main/images/logo.png" width="150" alt="catllm Logo">
       </td>
    </tr>
</table>

# cat-llm

A Python package for categorizing text data and images using Large Language Models (LLMs) and vision models. Designed for social science research and survey data analysis.

[![PyPI version](https://badge.fury.io/py/cat-llm.svg)](https://badge.fury.io/py/cat-llm)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

`catllm` provides tools for automated text classification using state-of-the-art language models. This package was developed as part of research comparing different LLM approaches for coding open-ended survey responses, with applications in social science research, survey analysis, and qualitative data processing.

## Academic Research

This package implements the methodology described in our research paper:

**"Can social scientists use large language models (LLMs) to code open-ended survey responses across complexity levels"**

Our study uses the UC Berkeley Social Networks Study as a test case, comparing GPT-4o, Claude Sonnet 3.7, Llama 3.1 variant Sonnet Large, and Mistral Large against human annotators. Key findings include:

- Proprietary models (GPT-4o, Claude) achieve **97% accuracy** on straightforward questions and **88-91%** on complex interpretive tasks
- Open-source models achieve **95-96% accuracy** on straightforward questions and up to **87%** on complex tasks
- Response brevity (under 50 characters) improves classification accuracy by **7-11%**
- Minimal demographic variation in classification accuracy across population segments

### Academic Examples

The `academic_examples/` directory contains the exact methodology and code used in our research:

academic_examples/
├── ucnets_replication.py # Main replication script
├── model_comparison.py # Multi-model comparison analysis
├── performance_analysis.py # Accuracy and demographic analysis
├── methodology_documentation.md # Detailed methodology
└── README.md # Academic usage instructions

text

**Note**: The academic examples require access to the UC Berkeley Social Networks Study dataset. Contact [chrissoria@berkeley.edu] for data access requests or collaboration opportunities.

## Installation

pip install cat-llm

text

## Quick Start

import catllm
import os

Basic text categorization
categories = catllm.explore_corpus(
survey_question="What motivates you most at work?",
survey_input=["flexible schedule", "good pay", "interesting projects"],
api_key=os.environ.get("OPENAI_API_KEY"),
cat_num=5,
divisions=10
)

Categorize individual responses
result = catllm.categorize_text(
text="I love the flexibility and remote work options",
categories=["compensation", "flexibility", "growth", "culture"],
api_key=os.environ.get("OPENAI_API_KEY")
)

text

## Supported Models

- **OpenAI**: GPT-4o, GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude Sonnet 3.7, Claude Haiku  
- **Open Source**: Llama 3.1, Mistral Large (via API)

## Features

- **Multi-model support**: Compare performance across different LLMs
- **Batch processing**: Efficiently process large survey datasets
- **Demographic analysis**: Analyze classification performance across population segments
- **Response optimization**: Built-in tools for optimizing response length and format
- **Academic reproducibility**: Exact replication of peer-reviewed methodology

## Documentation

- [Full Documentation](https://github.com/chrissoria/cat-llm#readme)
- [API Reference](https://github.com/chrissoria/cat-llm/docs)
- [Examples](https://github.com/chrissoria/cat-llm/tree/main/src/catllm/examples)

## Research Applications

This package is particularly suited for:

- **Social science research**: Survey response coding and analysis
- **Market research**: Customer feedback categorization
- **Qualitative analysis**: Systematic coding of open-ended responses
- **Educational research**: Student response analysis
- **Policy research**: Public opinion analysis

## Citation

If you use this package in your research, please cite:

@article{soria2025llm,
title={Can social scientists use large language models (LLMs) to code open-ended survey responses across complexity levels},
author={Soria, Christopher},
journal={[Journal Name]},
year={2025},
publisher={[Publisher]},
url={[Paper URL when published]}
}

@software{soria2025catllm,
title={cat-llm: A Python package for LLM-based text categorization},
author={Soria, Christopher},
year={2025},
url={https://github.com/chrissoria/cat-llm},
version={0.0.8}
}

text

## Best Practices (Based on Research Findings)

1. **Design concise survey questions** that elicit 10-50 character responses
2. **Implement human-in-the-loop review** for complex interpretive tasks  
3. **Select models based on task complexity**: Use proprietary models for nuanced tasks
4. **Monitor demographic performance**: Ensure consistent accuracy across population segments
5. **Validate narratives**: Check that models don't produce unintended social interpretations

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Christopher Soria
- **Email**: chrissoria@berkeley.edu
- **Institution**: UC Berkeley
- **GitHub**: [@chrissoria](https://github.com/chrissoria)

## Acknowledgments

This research was conducted using the UC Berkeley Social Networks Study. We thank the participants and research team for making this work possible.