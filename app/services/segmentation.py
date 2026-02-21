"""Wall segmentation service using SegFormer"""
import os
import cv2
import numpy as np
from app.core.config import get_settings

settings = get_settings()

# Configure HuggingFace cache
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_hf_cache = os.path.join(_project_root, ".cache", "huggingface")
os.makedirs(_hf_cache, exist_ok=True)
os.environ.setdefault("HF_HOME", _hf_cache)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _hf_cache)

# ADE20K class index for wall
WALL_CLASS_ID = 0

# Lazy-loaded model
_segmentation_model = None
_segmentation_processor = None


def get_segmentation_model():
    """Lazy load SegFormer model for wall detection"""
    global _segmentation_model, _segmentation_processor

    if _segmentation_model is None:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        import torch

        model_name = settings.model_name
        hf_token = os.environ.get("HF_TOKEN")

        print(f"🤖 Loading SegFormer model ({model_name})...")

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            
            _segmentation_processor = SegformerImageProcessor.from_pretrained(
                model_name,
                token=hf_token
            )
            _segmentation_model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                token=hf_token
            )
        
        _segmentation_model.eval()

        # Use CUDA if available, else MPS (Apple Silicon), else CPU
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        _segmentation_model.to(device)
        print(f"✓ Model loaded on {device}")

    return _segmentation_model, _segmentation_processor


def warmup_model():
    """Warmup model with dummy inference"""
    import torch
    from PIL import Image

    model, processor = get_segmentation_model()

    print("🔥 Warming up model...")
    dummy_img = Image.new('RGB', (256, 256), color='white')
    device = next(model.parameters()).device

    inputs = processor(images=dummy_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        h_orig, w_orig = dummy_img.size

        logits_resized = torch.nn.functional.interpolate(
            logits, size=(h_orig, w_orig), mode='bilinear', align_corners=False
        )
        preds = logits_resized.argmax(dim=1).squeeze(0).cpu().numpy()

    print("✓ Warmup complete")


def get_wall_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extract wall mask using SegFormer with guided filter refinement

    Args:
        img_rgb: RGB image as numpy array

    Returns:
        Binary mask (255 = wall, 0 = not wall) at original image size
    """
    import torch
    from PIL import Image

    model, processor = get_segmentation_model()
    device = next(model.parameters()).device

    h_orig, w_orig = img_rgb.shape[:2]

    # Convert to PIL (already uint8 from read_image)
    img_pil = Image.fromarray(img_rgb)

    # Calculate optimal processing size
    # SegFormer is trained on 512x512. Going too high (e.g. 1024 or 2048) increases RAM usage 
    # exponentially and causes the OS to kill the process (OOM) on weak VPS servers.
    # 512 is the safest maximum that preserves enough edges for guided filtering.
    max_size = 512
    scale = min(max_size / w_orig, max_size / h_orig)
    if scale < 1.0:
        proc_w, proc_h = int(w_orig * scale), int(h_orig * scale)
    else:
        proc_w, proc_h = w_orig, h_orig

    # Ensure dimensions are multiples of 32
    proc_w = (proc_w // 32) * 32
    proc_h = (proc_h // 32) * 32

    # Process image
    inputs = processor(
        images=img_pil,
        return_tensors="pt",
        size={"height": proc_h, "width": proc_w}
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits and resize to original size
    logits = outputs.logits

    # VERY IMPORTANT: To avoid Out-Of-Memory (OOM) errors on large images,
    # we DO NOT interpolate the entire 150-class logits tensor (which can take 7GB+ of RAM).
    # Instead, we take the argmax at the low resolution first, create a 1-channel mask,
    # and then interpolate only that 1-channel mask to the original size.
    
    # 1. Take argmax at low resolution
    preds_low = logits.argmax(dim=1)  # Shape: [1, H_low, W_low]
    
    # 2. Create binary mask for wall class
    wall_mask_low = (preds_low == WALL_CLASS_ID).float().unsqueeze(1)  # Shape: [1, 1, H_low, W_low]
    
    # Free memory
    del outputs
    del logits
    del preds_low
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. Interpolate the 1-channel binary mask to original size
    wall_mask_resized = torch.nn.functional.interpolate(
        wall_mask_low, size=(h_orig, w_orig), mode='bilinear', align_corners=False
    )
    
    # 4. Convert back to numpy and threshold
    mask_np = wall_mask_resized.squeeze().cpu().numpy()
    base_mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255

    # Clean up the mask using Connected Components Analysis
    # Walls are massive structures. We can safely remove any isolated blobs
    # (hallucinations on the ceiling, furniture) that are smaller than 0.5% of the image.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(base_mask_uint8, connectivity=8)
    
    img_area = h_orig * w_orig
    min_area = img_area * 0.005  # 0.5% minimum area
    
    cleaned_mask = np.zeros_like(base_mask_uint8)
    for i in range(1, num_labels): # Skip 0 (background)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == i] = 255
            
    # Close any small holes inside the legitimate wall components
    kernel_close = np.ones((7, 7), np.uint8)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # We pass the cleaned mask exactly as is (no blind dilation) to the Guided Filter.
    # This ensures no bleeding onto baseboards or molding.
    base_mask_f = cleaned_mask.astype(np.float32) / 255.0

    # Apply Guided Filter for edge refinement
    try:
        # We use a balanced radius (12) for the guided filter to perfectly
        # align the upscaled edges with the physical edges in the high-res image.
        radius = 12
        eps = 1e-5

        guide = img_rgb.astype(np.float32) / 255.0

        refined_mask = cv2.ximgproc.guidedFilter(
            guide=guide,
            src=base_mask_f,
            radius=radius,
            eps=eps
        )

        # Smooth and constrain values
        refined_mask = np.clip(refined_mask, 0.0, 1.0)
        wall_mask_final = (refined_mask * 255.0).astype(np.uint8)
    except Exception as e:
        print(f"⚠ Guided filter failed, using raw mask: {e}")
        wall_mask_final = cleaned_mask

    return wall_mask_final
