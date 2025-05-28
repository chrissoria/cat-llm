# images/__init__.py
"""Image resources for catllm package."""

from importlib import resources
import catllm.images

AVAILABLE_IMAGES = [
    'logo.png',
    # Add other image names here
]

def load_image(image_name):
    """Load an image from the package."""
    if image_name not in AVAILABLE_IMAGES:
        raise ValueError(f"Image '{image_name}' not found. Available: {AVAILABLE_IMAGES}")
    
    try:
        with resources.files(catllm.images).joinpath(image_name).open('rb') as f:
            return f.read()
    except AttributeError:
        # Fallback for older Python versions
        with resources.open_binary(catllm.images, image_name) as f:
            return f.read()
