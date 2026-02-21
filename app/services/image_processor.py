"""Image processing service with caching"""
import os
import threading
import cv2
import numpy as np
import hashlib
from datetime import datetime
from typing import Optional, Tuple
from app.core.config import get_settings
from app.core.cache import cache
from .segmentation import get_wall_mask

settings = get_settings()

# Limit to 1 concurrent SegFormer inference to prevent OOM on low-RAM servers (4 GB)
_process_semaphore = threading.Semaphore(1)


def get_image_hash(img_path: str) -> str:
    """Generate hash of image file for caching"""
    with open(img_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def read_image(img_path: str, max_dim: int = 1920) -> np.ndarray:
    """Read image, convert to RGB, and resize if too large"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    # Downscale image if it's too large to prevent OOM
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_colored_image(img: np.ndarray, new_color: Optional[Tuple[int, int, int]] = None,
                      pattern_image: Optional[str] = None) -> np.ndarray:
    """
    Apply color or pattern to image with realistic lighting preservation

    Args:
        img: Input RGB image
        new_color: RGB tuple for solid color
        pattern_image: Path to pattern image

    Returns:
        Colored RGB image
    """
    if new_color is not None:
        # Extract luminance from original image using LAB color space
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_channel, a_channel, b_channel = cv2.split(img_lab)

        # Create solid color image
        color_img = np.zeros_like(img, dtype=np.uint8)
        color_img[:] = new_color

        # Convert new color to LAB
        color_lab = cv2.cvtColor(color_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        _, new_a, new_b = cv2.split(color_lab)

        # Combine original luminance with new color channels
        final_lab = cv2.merge([l_channel, new_a, new_b])
        final_lab = np.clip(final_lab, 0, 255).astype(np.uint8)

        new_rgb_image = cv2.cvtColor(final_lab, cv2.COLOR_LAB2RGB)

        # Multiply blend for realistic paint appearance
        img_f = img.astype(np.float32) / 255.0
        color_f = color_img.astype(np.float32) / 255.0
        multiply_blend = img_f * color_f

        # Mix LAB and multiply blend for best results
        mixed = cv2.addWeighted(
            (multiply_blend * 255).astype(np.uint8), 0.3,
            new_rgb_image, 0.7, 0
        )
        return mixed

    elif pattern_image is not None:
        pattern = cv2.imread(pattern_image)
        if pattern is None:
            raise ValueError(f"Could not read pattern at {pattern_image}")
        pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2RGB)

        # Tile pattern to match image size
        h, w = img.shape[:2]
        ph, pw = pattern.shape[:2]

        tiled_pattern = np.zeros_like(img)
        for i in range(0, h, ph):
            for j in range(0, w, pw):
                ch = min(ph, h - i)
                cw = min(pw, w - j)
                tiled_pattern[i:i+ch, j:j+cw] = pattern[0:ch, 0:cw]

        # Blend pattern with original luminance
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        pattern_hsv = cv2.cvtColor(tiled_pattern, cv2.COLOR_RGB2HSV)

        h_p, s_p, v_p = cv2.split(pattern_hsv)
        h_i, s_i, v_i = cv2.split(img_hsv)

        # Keep original brightness
        new_hsv = cv2.merge([h_p, s_p, v_i])
        return cv2.cvtColor(new_hsv, cv2.COLOR_HSV2RGB)

    return img


def merge_images(img: np.ndarray, colored_image: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    """Apply new color/pattern only to wall pixels"""
    # Ensure mask dimensions match the image (in case of cached older masks or downscaling)
    if wall_mask.shape[:2] != img.shape[:2]:
        wall_mask = cv2.resize(wall_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_f = wall_mask.astype(np.float32) / 255.0
    if len(mask_f.shape) == 2:
        mask_f = cv2.merge([mask_f, mask_f, mask_f])

    img_f = img.astype(np.float32)
    colored_f = colored_image.astype(np.float32)

    final_f = colored_f * mask_f + img_f * (1.0 - mask_f)
    return np.clip(final_f, 0, 255).astype(np.uint8)


def process_image(input_path: str, output_path: str,
                 new_color: Optional[Tuple[int, int, int]] = None,
                 pattern_path: Optional[str] = None) -> Tuple[str, float]:
    """
    Process image with wall color/pattern change

    Args:
        input_path: Path to input image
        output_path: Path to save output
        new_color: RGB tuple for solid color
        pattern_path: Path to pattern image

    Returns:
        Tuple of (output_path, processing_time)
    """
    with _process_semaphore:
        return _process_image_impl(input_path, output_path, new_color, pattern_path)


def _process_image_impl(input_path: str, output_path: str,
                        new_color: Optional[Tuple[int, int, int]] = None,
                        pattern_path: Optional[str] = None) -> Tuple[str, float]:
    """Internal implementation (called under semaphore)"""
    start_time = datetime.now()

    # Read image
    img = read_image(input_path)

    # Get image hash for caching
    img_hash = get_image_hash(input_path)

    # Try to get cached wall mask (include model name and version to invalidate old caches)
    model_safe_name = settings.model_name.replace("/", "_")
    cache_key = f"wall_mask_{model_safe_name}_v3:{img_hash}"
    wall_mask = cache.get(cache_key)

    if wall_mask is None:
        print(f"🔍 Extracting wall mask with SegFormer (not cached)...")
        wall_mask = get_wall_mask(img)

        # Cache the mask for future use
        cache.set(cache_key, wall_mask, ttl=settings.mask_cache_ttl)
        print(f"✓ Wall mask cached")
    else:
        print(f"⚡ Using cached wall mask")

    # Apply color/pattern
    print(f"🎨 Applying color transformation...")
    colored_image = get_colored_image(img, new_color, pattern_path)

    # Merge images
    print(f"🔀 Merging images...")
    final_img = merge_images(img, colored_image, wall_mask)

    # Save image
    final_img_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, final_img_bgr)

    processing_time = (datetime.now() - start_time).total_seconds()
    print(f"✓ Done! Processing time: {processing_time:.2f}s")

    return output_path, processing_time


def preload_mask_for_image(img_path: str) -> bool:
    """
    Preload and cache wall mask for an image at startup.
    When users later select this image, the mask will already be in cache = instant processing.
    Returns True on success.
    """
    with _process_semaphore:
        try:
            img = read_image(img_path)
            img_hash = get_image_hash(img_path)
            model_safe_name = settings.model_name.replace("/", "_")
            cache_key = f"wall_mask_{model_safe_name}_v3:{img_hash}"
            if cache.get(cache_key) is not None:
                return True  # already cached
            wall_mask = get_wall_mask(img)
            cache.set(cache_key, wall_mask, ttl=settings.mask_cache_ttl)
            return True
        except Exception as e:
            print(f"⚠ Preload failed for {img_path}: {e}")
            return False
