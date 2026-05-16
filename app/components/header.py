"""
App header — minimal. Renders only the Colophon expander.

(The full masthead was removed per design feedback; theme styling now carries
the brand on its own.)
"""

import streamlit as st


def render_header():
    """Render the Colophon expander."""
    with st.expander("Colophon & Notes"):
        st.markdown(_colophon_md)


_colophon_md = """
**Privacy notice.** When using cloud LLM providers, your data is transmitted
to third-party APIs. Do not upload sensitive, confidential, or personally
identifiable information.

---

**CatLLM** is an open-source ecosystem for processing text, image, and PDF
data using Large Language Models — a domain-aware apparatus for the
cataloguing of unstructured response.

### Domains
- *General* — domain-neutral text analysis
- *Survey* — survey response classification with survey-framed prompts
- *Social Media* — fetch and classify posts from Threads, Bluesky, Reddit, Mastodon, YouTube
- *Academic* — fetch and classify papers from OpenAlex
- *Policy* — classify legislative & policy documents from registered sources
- *Web* — classify content fetched from URLs
- *Cognitive* — CERAD cognitive assessment drawing scoring

### Functions
- *Classify* — assign categories (manual or auto-extracted) to text / PDF / images
- *Extract* — discover categories from your data
- *Explore* — saturation analysis for category discovery
- *Summarize* — generate concise summaries

### Links
- **PyPI** — [pip install cat-llm](https://pypi.org/project/cat-llm/)
- **GitHub** — [github.com/chrissoria/cat-llm](https://github.com/chrissoria/cat-llm)

### Citation
```
Soria, C. (2025). CatLLM: A Python package for LLM-based text classification.
DOI: 10.5281/zenodo.15532316
```
"""
