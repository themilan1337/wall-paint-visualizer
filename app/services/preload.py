"""Preload wall masks for demo images at startup"""
import os
from app.core.config import get_settings
from app.services.image_processor import preload_mask_for_image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def preload_images() -> int:
    """
    Preload wall masks for all images in preload folder.
    Masks are cached so when users select these images, processing is instant.
    Returns number of successfully preloaded images.
    """
    settings = get_settings()
    preload_dir = settings.preload_folder

    if not os.path.isdir(preload_dir):
        print(f"⚠ Preload folder not found: {preload_dir}")
        return 0

    files = [
        f for f in os.listdir(preload_dir)
        if '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    ]
    files.sort()

    if not files:
        print("⚠ No images in preload folder")
        return 0

    print(f"\n🖼 Preloading {len(files)} demo images...")
    success = 0
    for filename in files:
        path = os.path.join(preload_dir, filename)
        if preload_mask_for_image(path):
            success += 1
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename}")

    print(f"✓ Preloaded {success}/{len(files)} images")
    return success
