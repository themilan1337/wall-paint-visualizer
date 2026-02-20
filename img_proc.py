import os
import cv2
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use project-local cache for HuggingFace (must be set before importing transformers)
_project_root = os.path.dirname(os.path.abspath(__file__))
_hf_cache = os.path.join(_project_root, ".cache", "huggingface")
os.makedirs(_hf_cache, exist_ok=True)
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = _hf_cache
if "HUGGINGFACE_HUB_CACHE" not in os.environ:
    os.environ["HUGGINGFACE_HUB_CACHE"] = _hf_cache

# ADE20K class index for wall (semantic segmentation)
WALL_CLASS_ID = 0

# Lazy-loaded SegFormer model (avoids loading on import)
_segmentation_model = None
_segmentation_processor = None


def _get_segmentation_model():
    """Lazy load SegFormer model for wall detection."""
    global _segmentation_model, _segmentation_processor
    if _segmentation_model is None:
        try:
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            import torch

            # Using a larger, much more accurate model
            model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
            
            # Fetch token from environment, required for higher rate limits on HF Hub
            hf_token = os.environ.get("HF_TOKEN")
            
            print(f"Loading SegFormer model ({model_name}). This might take a while on first run...")
            
            _segmentation_processor = SegformerImageProcessor.from_pretrained(
                model_name, 
                token=hf_token
            )
            _segmentation_model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                token=hf_token
            )
            _segmentation_model.eval()
            _segmentation_model.to("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            print("Model loaded successfully!")
        except Exception as e:
            print(f"SegFormer model load failed: {e}")
            raise
    return _segmentation_model, _segmentation_processor

def warmup_model():
    """Load model into memory and run a dummy inference to warm up the pipeline."""
    try:
        model, processor = _get_segmentation_model()
        
        # Run dummy inference to compile/warm up
        import torch
        from PIL import Image
        print("Running dummy inference for warmup...")
        dummy_img = Image.new('RGB', (256, 256), color='white')
        device = next(model.parameters()).device
        
        inputs = processor(images=dummy_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # logits: [batch, num_classes, height, width]
        logits = outputs.logits
        # Resize logits to original image size before argmax for better precision
        h_orig, w_orig = dummy_img.size
        
        # Interpolate logits to original image size
        logits_resized = torch.nn.functional.interpolate(
            logits, size=(h_orig, w_orig), mode='bilinear', align_corners=False
        )
        
        preds = logits_resized.argmax(dim=1).squeeze(0).cpu().numpy()
        
        print("Warmup complete. Ready for real requests.")
    except Exception as e:
        print(f"Warmup failed: {e}")


def getWallMaskSegFormer(img_rgb):
    """
    Use SegFormer semantic segmentation to extract wall pixels.
    Uses Guided Filter to refine the mask edges perfectly to the image objects.
    Returns binary mask (255 = wall, 0 = not wall) at original image size.
    """
    import torch
    from PIL import Image

    model, processor = _get_segmentation_model()
    device = next(model.parameters()).device

    # Convert BGR/RGB to PIL
    img_pil = Image.fromarray(img_rgb.astype(np.uint8))

    # Process and run inference
    inputs = processor(images=img_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # logits: [batch, num_classes, height, width]
    logits = outputs.logits
    # Resize logits to original image size before argmax for better precision
    h_orig, w_orig = img_rgb.shape[:2]
    
    # Interpolate logits to original image size
    logits_resized = torch.nn.functional.interpolate(
        logits, size=(h_orig, w_orig), mode='bilinear', align_corners=False
    )
    
    preds = logits_resized.argmax(dim=1).squeeze(0).cpu().numpy()

    # Wall = class 0 in ADE20K
    wall_mask = (preds == WALL_CLASS_ID).astype(np.float32)

    # REFINEMENT: Use Guided Filter to align mask edges perfectly to the high-res image
    try:
        # Guided filter parameters
        radius = 20
        eps = 1e-4
        
        # Guide image should be normalized [0, 1] float32
        guide = img_rgb.astype(np.float32) / 255.0
        
        # Apply Guided Filter
        refined_mask = cv2.ximgproc.guidedFilter(
            guide=guide, 
            src=wall_mask, 
            radius=radius, 
            eps=eps
        )
        
        # Threshold back to binary
        wall_mask_final = (refined_mask > 0.5).astype(np.uint8) * 255
    except Exception as e:
        print(f"Guided filter failed, falling back to raw mask: {e}")
        wall_mask_final = (wall_mask * 255).astype(np.uint8)

    return wall_mask_final


def readImage(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def getColoredImage(img, new_color, pattern_image=None):
    """
    Apply color or pattern to the image in a realistic way,
    preserving lighting, shadows and luminance.
    """
    if new_color is not None:
        # Extract luminance from original image
        # Convert to LAB color space
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        
        # Create a solid color image with the new color
        color_img = np.zeros_like(img, dtype=np.uint8)
        color_img[:] = new_color
        
        # Convert new color image to LAB
        color_lab = cv2.cvtColor(color_img, cv2.COLOR_RGB2LAB).astype(np.float32)
        _, new_a, new_b = cv2.split(color_lab)
        
        # Combine original Luminance with new A and B channels
        # Reduce the intensity of the new color slightly so shadows still look natural
        final_lab = cv2.merge([l_channel, new_a, new_b])
        final_lab = np.clip(final_lab, 0, 255).astype(np.uint8)
        
        new_rgb_image = cv2.cvtColor(final_lab, cv2.COLOR_LAB2RGB)
        
        # Alternative approach: Multiply blend for more realistic paint look
        img_f = img.astype(np.float32) / 255.0
        color_f = color_img.astype(np.float32) / 255.0
        # Multiply blend preserves shadows better
        multiply_blend = img_f * color_f
        # Overlay blend
        overlay_blend = np.where(img_f < 0.5, 2 * img_f * color_f, 1 - 2 * (1 - img_f) * (1 - color_f))
        
        # Mix LAB method and multiply/overlay for best realistic paint appearance
        mixed = cv2.addWeighted((multiply_blend * 255).astype(np.uint8), 0.3, new_rgb_image, 0.7, 0)
        return mixed

    elif pattern_image is not None:
        pattern = cv2.imread(pattern_image)
        if pattern is None:
            raise ValueError(f"Could not read pattern at {pattern_image}")
        pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2RGB)
        
        # Tile or resize the pattern to match image size
        h, w = img.shape[:2]
        ph, pw = pattern.shape[:2]
        
        # Tiling the pattern
        tiled_pattern = np.zeros_like(img)
        for i in range(0, h, ph):
            for j in range(0, w, pw):
                ch = min(ph, h - i)
                cw = min(pw, w - j)
                tiled_pattern[i:i+ch, j:j+cw] = pattern[0:ch, 0:cw]
                
        # Blend pattern with original image's luminance
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        pattern_hsv = cv2.cvtColor(tiled_pattern, cv2.COLOR_RGB2HSV)
        
        h_p, s_p, v_p = cv2.split(pattern_hsv)
        h_i, s_i, v_i = cv2.split(img_hsv)
        
        # Keep original brightness (v_i)
        new_hsv = cv2.merge([h_p, s_p, v_i])
        return cv2.cvtColor(new_hsv, cv2.COLOR_HSV2RGB)
        
    return img


def mergeImages(img, colored_image, wall_mask):
    """Apply new color/pattern only to wall pixels (where wall_mask == 255)."""
    # Soft blending with mask
    mask_f = wall_mask.astype(np.float32) / 255.0
    if len(mask_f.shape) == 2:
        mask_f = cv2.merge([mask_f, mask_f, mask_f])
        
    img_f = img.astype(np.float32)
    colored_f = colored_image.astype(np.float32)
    
    final_f = colored_f * mask_f + img_f * (1.0 - mask_f)
    return np.clip(final_f, 0, 255).astype(np.uint8)


def changeColor(input_path, output_path, new_color=None, pattern_path=None):
    start = datetime.timestamp(datetime.now())
    img = readImage(input_path)
    
    print(f"Processing image {input_path}...")
    
    colored_image = getColoredImage(img, new_color, pattern_path)
    
    print("Extracting wall mask with SegFormer...")
    wall_mask = getWallMaskSegFormer(img)
    
    print("Merging images...")
    final_img = mergeImages(img, colored_image, wall_mask)
    
    # Save image
    final_img_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, final_img_bgr)
    
    end = datetime.timestamp(datetime.now())
    print(f"Done! Total processing time: {end - start:.2f}s")
    return output_path

