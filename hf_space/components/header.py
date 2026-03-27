"""
App header: logo, title, about section.
"""

import os
import streamlit as st


def render_header():
    """Render the CatLLM logo, title, and about expander."""
    logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "logo.png")

    col_logo, col_title = st.columns([1, 6])
    with col_logo:
        if os.path.exists(logo_path):
            st.image(logo_path, width=100)
    with col_title:
        st.title("CatLLM")
        st.markdown("LLM-powered classification, extraction, and summarization across domains.")

    with st.expander("About This App"):
        st.markdown("""
**Privacy Notice:** When using cloud LLM providers, your data is sent to third-party APIs.
Do not upload sensitive, confidential, or personally identifiable information (PII).

---

**CatLLM** is an open-source ecosystem for processing text, image, and PDF data using Large Language Models.

### Domains
- **General** -- domain-neutral text analysis
- **Survey** -- survey response classification with survey-framed prompts
- **Social Media** -- fetch and classify posts from Threads, Bluesky, Reddit, Mastodon, YouTube
- **Academic** -- fetch and classify papers from OpenAlex
- **Policy** -- classify legislative/policy documents from registered sources
- **Web** -- classify content fetched from URLs
- **Cognitive** -- CERAD cognitive assessment drawing scoring

### Functions
- **Classify** -- assign categories (manual or auto-extracted) to text/PDF/images
- **Extract** -- discover categories from your data
- **Explore** -- saturation analysis for category discovery
- **Summarize** -- generate concise summaries

### Links
- **PyPI**: [pip install cat-llm](https://pypi.org/project/cat-llm/)
- **GitHub**: [github.com/chrissoria/cat-llm](https://github.com/chrissoria/cat-llm)

### Citation
```
Soria, C. (2025). CatLLM: A Python package for LLM-based text classification. DOI: 10.5281/zenodo.15532316
```
""")
