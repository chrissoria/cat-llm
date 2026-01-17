#!/usr/bin/env python
"""
Test the extract() function with image input across all vision-capable providers.
"""

import sys
import os

# Add the src directory to path
src_path = '/Users/chrissoria/Documents/Research/cat-llm/src'
sys.path.insert(0, src_path)

# Clear cached modules
modules_to_remove = [key for key in sys.modules.keys() if key.startswith('catllm')]
for mod in modules_to_remove:
    del sys.modules[mod]

from dotenv import load_dotenv
load_dotenv('/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env')

api_keys = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY"),
    "xai": os.getenv("XAI_API_KEY"),
}

print("=" * 70)
print("Test: extract() with input_type='image'")
print("=" * 70)

# Create test image if needed
test_image = "/Users/chrissoria/Documents/Research/cat-llm/tests/test_image.png"
if not os.path.exists(test_image):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 180, 180], outline='blue', width=3)
    draw.ellipse([50, 50, 150, 150], outline='red', width=2)
    draw.text((60, 90), "Test", fill='black')
    img.save(test_image)
    print(f"Created test image: {test_image}")

# Import the extract function
from catllm import extract

# Vision-capable models
vision_models = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "google": "gemini-2.0-flash",
    "mistral": "pixtral-12b-2409",
    "xai": "grok-2-vision-1212",
}

results = []

for provider, model in vision_models.items():
    print(f"\n{'-' * 50}")
    print(f"Provider: {provider} | Model: {model}")
    print(f"{'-' * 50}")

    if not api_keys.get(provider):
        print("  SKIPPED - No API key")
        results.append({"provider": provider, "success": None, "skipped": True})
        continue

    try:
        result = extract(
            input_data=test_image,
            api_key=api_keys[provider],
            input_type="image",
            description="simple test image with geometric shapes",
            max_categories=5,
            categories_per_chunk=5,
            divisions=1,
            user_model=model,
            creativity=0.3,
            specificity="broad",
            model_source=provider,
            iterations=1
        )

        if result and result.get('top_categories'):
            cats = result['top_categories'][:3]
            print(f"  Categories: {cats}")
            print(f"  PASS")
            results.append({"provider": provider, "success": True, "skipped": False})
        else:
            print(f"  FAIL - No categories extracted")
            results.append({"provider": provider, "success": False, "skipped": False})
    except Exception as e:
        print(f"  ERROR: {str(e)[:80]}")
        results.append({"provider": provider, "success": False, "skipped": False, "error": str(e)})

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
    print(f"\nFailed:")
    for r in results:
        if r.get('success') is False:
            print(f"  - {r['provider']}: {r.get('error', 'No output')[:50]}")

sys.exit(1 if failed > 0 else 0)
