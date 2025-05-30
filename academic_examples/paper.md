---
title: 'CatLLM: A Python package for Generating, Assigning, and Scoring Open-Ended Survey Data and Images'
tags:
  - Python
  - social science
  - demography
  - sociology
  - content coding
authors:
  - name: Chris Soria
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: University of California, Berkeley, United States
   index: 1
date: 29 May 2025
bibliography: catllm.bib
editor_options: 
  markdown: 
    wrap: 72
---

# Summary

The rapid advancement of large language models (LLMs) and vision models
has opened new possibilities for automated text and image analysis in
social science research. Researchers increasingly seek to leverage these
powerful tools for tasks such as coding open-ended survey responses,
categorizing qualitative data, and analyzing visual content at scale.
However, the field faces significant reproducibility challenges due to
inconsistent output formatting across different models, varying API
interfaces, and the lack of standardized workflows for integrating LLM
outputs into traditional statistical analysis pipelines. The majority of
research applications require systematic approaches to ensure
consistent, structured data extraction that can be readily analyzed
using established quantitative methods.

# Statement of need

`CatLLM` is a Python package designed for researchers conducting text and
image analysis using large language and vision models. The package was
designed to provide a standardized, function-based interface to common
research operations such as categorical text extraction, corpus-level
analysis, multi-feature image classification, and quality assessment
workflows.

`CatLLM` incorporates several essential features for academic research
workflows. The package includes built-in validation mechanisms to ensure
user-entered categories function as intended, preventing common
specification errors that can compromise analysis quality. Comprehensive
error handling protects researchers from mid-job failures that could
result in lost API costs and unsaved progress, while automatic progress
saving to local files provides additional data security. Cost-saving
functionality automatically skips NA rows rather than processing empty
entries, optimizing API usage efficiency. The package supports multiple
providers including OpenAI, Anthropic, Mistral, and Perplexity Llama
models, allowing researchers to select optimal services for their
specific analytical needs. Additionally, users can enhance task
specificity by inputting research hypotheses and detailed survey
question or image descriptions, enabling more contextually appropriate
model responses.

`CatLLM` integrates seamlessly with standard data science libraries
including pandas for data manipulation and provides direct CSV export
functionality for compatibility with statistical software packages. The
package was designed for social scientists, digital humanities
researchers, and students conducting qualitative-to-quantitative data
analysis. It has already been applied in research comparing LLM
performance across demographic groups using the UC Berkeley Social
Networks Study, demonstrating effectiveness in real-world survey
analysis scenarios. The package addresses critical reproducibility
concerns by standardizing output schemas regardless of underlying model
complexity or provider, enabling researchers to focus on analytical
insights rather than technical implementation details.

# Overview of Methods

The `CatLLM` package uses a binary matrix $M$ of shape
$(n_{\text{rows}}, n_{\text{categories}})$ where:

$i \in \{1, \dots, n_{\text{rows}}\}$ indexes rows (text objects or
images)

$j \in \{1, \dots, n_{\text{categories}}\}$ indexes categories

\begin{equation}\label{eq:binary_matrix}
M[i,j] = \begin{cases}
1 & \text{if the $i$-th object is annotated with category $j$} \\
0 & \text{otherwise}
\end{cases}
\end{equation}

This matrix enables systematic comparison across annotation methods with
standardized output regardless of the underlying approach.

For corpus-level theme identification, the package implements a chunked
extraction process. Given a corpus $C$ with $k$ random chunks and $n$
categories extracted per chunk:

\begin{equation}\label{eq:corpus_chunking}
T = \bigcup_{i=1}^{k} E(C_i, n)
\end{equation}

where $E(C_i, n)$ represents the extraction of $n$ themes from chunk
$C_i$, producing a total theme set $T$ of size $k \times n$ (with
potential duplicates).

\begin{equation}\label{eq:theme_frequency}
f(t) = \sum_{i=1}^{k} \sum_{j=1}^{n} \mathbf{1}_{t_{j}^{(i)} = t}
\end{equation}

where $f(t)$ represents the frequency of theme $t$ across all chunks,
and $\mathbf{1}_{t_{j}^{(i)} = t}$ is an indicator function that equals
1 when theme $j$ from chunk $i$ matches theme $t$.

The final output ranks unique themes by frequency:
$\{(t, f(t)) : t \in T\}$ sorted by $f(t)$ in descending order.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub
without a preferred citation) then you can do it with the example BibTeX
entry below for @fidgit.

For a quick reference, the following citation commands can be used: -
`@author:2001` -\> "Author et al. (2001)" - `[@author:2001]` -\>
"(Author et al., 2001)" - `[@author1:2001; @author2:2001]` -\> "(Author1
et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this: ![Caption for example
figure.](figure.png) and referenced from text using
\autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){width="20%"}

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and
Semyeong Oh, and support from Kathryn Johnston during the genesis of
this project.

# References
