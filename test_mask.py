import cv2
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

def test():
    img_path = 'public/images/img1.jpg'
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    
    img_pil = Image.fromarray(img_rgb)
    inputs = processor(images=img_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    h_orig, w_orig = img_rgb.shape[:2]
    logits_resized = torch.nn.functional.interpolate(
        logits, size=(h_orig, w_orig), mode='bilinear', align_corners=False
    )
    
    probs = torch.nn.functional.softmax(logits_resized, dim=1).squeeze(0)
    wall_prob = probs[0].cpu().numpy().astype(np.float32)
    
    guide = img_rgb.astype(np.float32) / 255.0
    
    # Test different params
    for r in [5, 8, 12]:
        for e in [1e-4, 1e-5, 1e-6]:
            refined = cv2.ximgproc.guidedFilter(guide=guide, src=wall_prob, radius=r, eps=e)
            mask = (refined > 0.5).astype(np.uint8) * 255
            cv2.imwrite(f'mask_r{r}_e{e}.png', mask)
            
if __name__ == '__main__':
    test()
