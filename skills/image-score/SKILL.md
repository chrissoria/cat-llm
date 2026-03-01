---
name: image-score
description: Score drawings or images against a reference description using vision models. Use for quality assessment, drawing scoring, and CERAD neuropsychological assessments.
---

# CatLLM Image Scoring

You are helping the user score images using CatLLM's vision model capabilities.

## Setup

```bash
pip install cat-llm
```

## Drawing Quality Scoring

Score images on a 1-5 scale based on similarity to a reference description:

```python
import catllm as cat

scores = cat.image_score_drawing(
    reference_image_description="A hand-drawn circle",
    image_input=["image1.jpg", "image2.jpg"],
    api_key="YOUR_API_KEY",
    user_model="gpt-4o",
    reference_image="reference.jpg",      # optional reference image
    filename="scores.csv",
    save_directory="./results",
)
```

Scoring scale: 1 (no similarity) to 5 (near-perfect match, 90%+).

## CERAD Drawing Assessment

Score drawings according to the official CERAD (Consortium to Establish a Registry for Alzheimer's Disease) scoring system:

```python
diamond_scores = cat.cerad_drawn_score(
    shape="diamond",                      # "circle", "diamond", "rectangles", "cube"
    image_input=df['image_paths'],
    api_key="YOUR_API_KEY",
    safety=True,                          # save intermediate results
    filename="diamond_scores.csv",
)
```

## Feature Extraction

Extract specific features from images:

```python
features = cat.image_features(
    image_description="Product photos",
    features_to_extract=["number of items", "primary color", "has price tag"],
    image_input="/path/to/images/",
    api_key="YOUR_API_KEY",
)
```

$ARGUMENTS
