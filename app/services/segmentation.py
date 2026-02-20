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

    # Convert to PIL
    img_pil = Image.fromarray(img_rgb.astype(np.uint8))

    # Calculate optimal processing size
    # Use higher resolution for better quality, but cap for performance
    max_size = 1024
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

    logits_resized = torch.nn.functional.interpolate(
        logits, size=(h_orig, w_orig), mode='bilinear', align_corners=False
    )

    # Use probability map for smoother edges
    probs = torch.nn.functional.softmax(logits_resized, dim=1).squeeze(0)
    wall_prob = probs[WALL_CLASS_ID].cpu().numpy().astype(np.float32)

    # Apply Guided Filter for edge refinement
    try:
        radius = 4
        eps = 1e-6

        guide = img_rgb.astype(np.float32) / 255.0

        refined_mask = cv2.ximgproc.guidedFilter(
            guide=guide,
            src=wall_prob,
            radius=radius,
            eps=eps
        )

        # Sharpen mask transitions
        refined_mask = np.clip((refined_mask - 0.5) * 10.0 + 0.5, 0, 1)

        wall_mask_final = (refined_mask * 255.0).astype(np.uint8)
    except Exception as e:
        print(f"⚠ Guided filter failed, using raw mask: {e}")
        wall_mask_final = (wall_prob > 0.5).astype(np.uint8) * 255

    return wall_mask_final
