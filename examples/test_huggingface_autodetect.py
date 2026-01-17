#!/usr/bin/env python
"""
Test script for HuggingFace endpoint auto-detection.

Tests two scenarios:
A. Model that works on generic endpoint (Qwen)
B. Model that works on Together endpoint (Llama 4)
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
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

if not huggingface_api_key:
    print("ERROR: HUGGINGFACE_API_KEY not found in environment")
    sys.exit(1)

print("=" * 70)
print("HuggingFace Endpoint Auto-Detection Test")
print("=" * 70)
print()

# Import the detection function
from catllm.text_functions import _detect_huggingface_endpoint

# Test scenarios
test_cases = [
    {
        "name": "Scenario A: Model that works on GENERIC endpoint",
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "expected_endpoint": "generic (/v1)",
        "expected_url_contains": "/v1",
        "expected_url_not_contains": "/together/",
    },
    {
        "name": "Scenario B: Model that works on TOGETHER endpoint",
        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "expected_endpoint": "together (/together/v1)",
        "expected_url_contains": "/together/v1",
        "expected_url_not_contains": None,
    },
]

results = []

for i, test in enumerate(test_cases, 1):
    print(f"\n{'-' * 70}")
    print(f"Test {i}: {test['name']}")
    print(f"{'-' * 70}")
    print(f"  Model: {test['model']}")
    print(f"  Expected: {test['expected_endpoint']}")
    print()
    print("  Detecting endpoint...")

    try:
        detected_url = _detect_huggingface_endpoint(huggingface_api_key, test['model'])
        print(f"  Detected URL: {detected_url}")

        # Check if detection was correct
        success = test['expected_url_contains'] in detected_url
        if test['expected_url_not_contains']:
            success = success and (test['expected_url_not_contains'] not in detected_url)

        if success:
            print(f"  Result: PASS - Correctly detected {test['expected_endpoint']}")
        else:
            print(f"  Result: FAIL - Expected {test['expected_endpoint']}, got {detected_url}")

        results.append({
            "test": test['name'],
            "model": test['model'],
            "detected_url": detected_url,
            "success": success,
        })

    except Exception as e:
        print(f"  Result: ERROR - {str(e)}")
        results.append({
            "test": test['name'],
            "model": test['model'],
            "detected_url": None,
            "success": False,
            "error": str(e),
        })

# Summary
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
passed = sum(1 for r in results if r['success'])
failed = len(results) - passed
print(f"  Passed: {passed}/{len(results)}")
print(f"  Failed: {failed}/{len(results)}")

if failed > 0:
    print("\nFailed tests:")
    for r in results:
        if not r['success']:
            error_msg = r.get('error', f"Got {r['detected_url']}")
            print(f"  - {r['model']}: {error_msg}")

print()

# Exit with appropriate code
sys.exit(0 if failed == 0 else 1)
