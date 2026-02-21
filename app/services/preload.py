"""Preload wall masks for demo images at startup"""
import gc
import os
import time
from app.core.config import get_settings
from app.services.image_processor import preload_mask_for_image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def preload_images() -> int:
    """
    Preload wall masks for demo images. Optimized for weak servers:
    - Limits number of images (preload_max_images)
    - GC + pause between images to avoid OOM
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

    max_n = settings.preload_max_images
    if max_n > 0:
        files = files[:max_n]

    if not files:
        print("⚠ No images in preload folder")
        return 0

    print(f"\n🖼 Preloading {len(files)} demo images (background)...")
    success = 0
    for i, filename in enumerate(files):
        path = os.path.join(preload_dir, filename)
        try:
            if preload_mask_for_image(path):
                success += 1
                print(f"  ✓ {filename}")
        except Exception as e:
            print(f"  ✗ {filename}: {e}")
        # Free memory and pause between images (gentle on weak servers)
        gc.collect()
        if i < len(files) - 1:
            time.sleep(2)

    print(f"✓ Preloaded {success}/{len(files)} images")
    return success
