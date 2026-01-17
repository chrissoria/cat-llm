#!/usr/bin/env python
"""
Test script to verify image functions work with all LLM providers.

Tests the explore_image_categories function with each supported provider using image mode.
"""

import sys
import os

# Add the src directory to path so we import from local code
src_path = '/Users/chrissoria/Documents/Research/cat-llm/src'
sys.path.insert(0, src_path)

# Clear any cached modules
modules_to_remove = [key for key in sys.modules.keys() if key.startswith('catllm')]
for mod in modules_to_remove:
    del sys.modules[mod]

from dotenv import load_dotenv

# Load API keys
load_dotenv('/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env')

# Get all API keys
api_keys = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY"),
    "xai": os.getenv("XAI_API_KEY"),
    "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
    "perplexity": os.getenv("PERPLEXITY_API_KEY"),
}

print("=" * 70)
print("Image Functions Test - explore_image_categories")
print("=" * 70)
print()

# Print API key status
print("API Keys Status:")
for provider, key in api_keys.items():
    status = "Found" if key else "MISSING"
    print(f"  {provider}: {status}")
print()

# Test image - use the title page PDF as an image source (or any test image)
# First, let's check for any test images in the tests directory
test_image = None
for ext in ['png', 'jpg', 'jpeg']:
    potential_path = f"/Users/chrissoria/Documents/Research/cat-llm/tests/test_image.{ext}"
    if os.path.exists(potential_path):
        test_image = potential_path
        break

# If no test image exists, create a simple one or use the title page
if test_image is None:
    # Create a simple test image
    try:
        from PIL import Image, ImageDraw
        test_image = "/Users/chrissoria/Documents/Research/cat-llm/tests/test_image.png"
        img = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 180, 180], outline='blue', width=3)
        draw.text((50, 90), "Test Image", fill='black')
        img.save(test_image)
        print(f"Created test image: {test_image}")
    except ImportError:
        print("Note: PIL not installed, skipping test image creation")
        print("Please create a test image at tests/test_image.png")
        sys.exit(1)

print(f"Test image: {test_image}")
print()

# Import the function
from catllm.image_functions import explore_image_categories

# Test configurations for each provider
# Using small/fast models and image mode to minimize cost and time
test_configs = [
    {
        "name": "OpenAI",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": api_keys["openai"],
    },
    {
        "name": "Anthropic",
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307",
        "api_key": api_keys["anthropic"],
    },
    {
        "name": "Google",
        "provider": "google",
        "model": "gemini-2.0-flash",
        "api_key": api_keys["google"],
    },
    {
        "name": "Mistral",
        "provider": "mistral",
        "model": "pixtral-12b-2409",  # Vision-capable Mistral model
        "api_key": api_keys["mistral"],
    },
    {
        "name": "xAI",
        "provider": "xai",
        "model": "grok-2-vision-1212",
        "api_key": api_keys["xai"],
    },
    {
        "name": "HuggingFace (Qwen2-VL - vision model)",
        "provider": "huggingface",
        "model": "Qwen/Qwen2-VL-72B-Instruct",
        "api_key": api_keys["huggingface"],
    },
]

results = []

for i, config in enumerate(test_configs, 1):
    print(f"\n{'-' * 70}")
    print(f"Test {i}/{len(test_configs)}: {config['name']}")
    print(f"{'-' * 70}")
    print(f"  Provider: {config['provider']}")
    print(f"  Model: {config['model']}")

    if not config['api_key']:
        print(f"  Result: SKIPPED - No API key")
        results.append({
            "name": config['name'],
            "provider": config['provider'],
            "model": config['model'],
            "success": None,
            "skipped": True,
            "error": "No API key",
        })
        continue

    # Skip non-vision models
    if config.get('skip_reason'):
        print(f"  Result: SKIPPED - {config['skip_reason']}")
        results.append({
            "name": config['name'],
            "provider": config['provider'],
            "model": config['model'],
            "success": None,
            "skipped": True,
            "error": config['skip_reason'],
        })
        continue

    print(f"  Testing (image mode)...")

    try:
        result = explore_image_categories(
            image_input=test_image,
            api_key=config['api_key'],
            image_description="simple test image with shapes and text",
            max_categories=5,
            categories_per_chunk=5,
            divisions=1,
            user_model=config['model'],
            creativity=0.3,
            specificity="broad",
            mode="image",
            model_source=config['provider'],
            iterations=1
        )

        if result and 'top_categories' in result and result['top_categories']:
            categories = result['top_categories'][:3]
            print(f"  Categories found: {categories}")
            print(f"  Result: PASS")
            results.append({
                "name": config['name'],
                "provider": config['provider'],
                "model": config['model'],
                "success": True,
                "skipped": False,
                "categories": categories,
            })
        else:
            print(f"  Result: FAIL - No categories extracted")
            results.append({
                "name": config['name'],
                "provider": config['provider'],
                "model": config['model'],
                "success": False,
                "skipped": False,
                "error": "No categories extracted",
            })

    except Exception as e:
        print(f"  Result: ERROR - {str(e)[:100]}")
        results.append({
            "name": config['name'],
            "provider": config['provider'],
            "model": config['model'],
            "success": False,
            "skipped": False,
            "error": str(e)[:200],
        })

# Summary
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")

passed = sum(1 for r in results if r.get('success') is True)
failed = sum(1 for r in results if r.get('success') is False)
skipped = sum(1 for r in results if r.get('skipped') is True)

print(f"  Passed:  {passed}")
print(f"  Failed:  {failed}")
print(f"  Skipped: {skipped}")
print(f"  Total:   {len(results)}")

if failed > 0:
    print(f"\nFailed tests:")
    for r in results:
        if r.get('success') is False:
            print(f"  - {r['name']}: {r.get('error', 'Unknown error')}")

if skipped > 0:
    print(f"\nSkipped tests:")
    for r in results:
        if r.get('skipped'):
            print(f"  - {r['name']}: {r.get('error', 'Unknown reason')}")

print()

# Exit with appropriate code
sys.exit(1 if failed > 0 else 0)
