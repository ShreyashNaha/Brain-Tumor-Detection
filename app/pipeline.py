"""
pipeline.py
───────────
Offline inference pipeline for Brain Tumor Detection.

Steps:
  1. Load & preprocess the uploaded MRI image
  2. Run EfficientNet-B0 inference → class + confidence scores
  3. IFFT-based image reconstruction (edge enhancement via freq-domain filtering)
  4. GradCAM heatmap overlay → tumor region marking
  5. Return original, enhanced, and overlay images as base64 strings
"""

import io
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn.functional as F
from torchvision import transforms
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

TUMOR_CLASSES = {'Glioma', 'Meningioma', 'Pituitary'}

# ── Preprocessing ──────────────────────────────────────────────────────────────
_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


def load_image(path_or_bytes) -> Image.Image:
    """Load image from file path or bytes object, convert to RGB."""
    if isinstance(path_or_bytes, str):
        img = Image.open(path_or_bytes)
    elif isinstance(path_or_bytes, bytes):
        img = Image.open(io.BytesIO(path_or_bytes))
    else:
        img = Image.open(path_or_bytes)
    return img.convert('RGB')


# ── IFFT Reconstruction ────────────────────────────────────────────────────────
def ifft_enhance(pil_img: Image.Image, boost: float = 1.35) -> Image.Image:
    """
    Subtle frequency-domain edge enhancement using FFT/IFFT.
    """
    img_np = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    h, w, c = img_np.shape
    enhanced = np.zeros_like(img_np)

    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_dist = np.sqrt(cy ** 2 + cx ** 2)

    sigma = max_dist * 0.25
    mask = 1.0 + (boost - 1.0) * (1 - np.exp(-(dist ** 2) / (2 * sigma ** 2)))

    for ch in range(c):
        spectrum  = fftshift(fft2(img_np[:, :, ch]))
        spectrum  = spectrum * mask
        restored  = np.real(ifft2(ifftshift(spectrum)))
        enhanced[:, :, ch] = np.clip(restored, 0, 1)

    enhanced_img = Image.fromarray((enhanced * 255).astype(np.uint8))
    enhanced_img = enhanced_img.filter(ImageFilter.UnsharpMask(radius=1, percent=40, threshold=2))
    return enhanced_img


# ── GradCAM ────────────────────────────────────────────────────────────────────
class GradCAM:
    """Lightweight GradCAM for timm EfficientNet models."""

    def __init__(self, model):
        self.model   = model
        self.grads   = None
        self.acts    = None
        self._hooks  = []
        self._register()

    def _register(self):
        target_layer = self.model.conv_head

        def fwd_hook(module, inp, out):
            self.acts = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.grads = grad_out[0].detach()

        self._hooks.append(target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(target_layer.register_full_backward_hook(bwd_hook))

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def apply_heatmap_overlay(original_pil: Image.Image,
                           cam: np.ndarray,
                           alpha_max: float = 0.55) -> Image.Image:
    """
    Blend GradCAM heatmap onto the MRI image using dynamic transparency.
    Draws a medical-style targeting box around the primary activation cluster.
    """
    base = original_pil.resize((IMG_SIZE, IMG_SIZE)).convert('RGBA')
    
    # Resize CAM to match image
    cam_rs = np.array(Image.fromarray((cam * 255).astype(np.uint8))
                      .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.float32) / 255.0

    # Medical Color Map (Jet-style)
    r = np.clip(cam_rs * 3,       0, 1)
    g = np.clip(cam_rs * 3 - 1,   0, 1)
    b = np.clip(cam_rs * 3 - 2,   0, 1)
    
    # Dynamic alpha: only color the "hot" zones, keep rest normal
    a = np.clip((cam_rs - 0.35) * 2.0, 0, 1) * alpha_max

    heat_rgba = np.stack([r, g, b, a], axis=-1)
    heat_img  = Image.fromarray((heat_rgba * 255).astype(np.uint8), 'RGBA')

    overlay = Image.alpha_composite(base, heat_img)

    # ── Draw Targeting Box ──
    draw = ImageDraw.Draw(overlay)
    
    # Threshold for finding the "core" of the model's focus (> 70% intensity)
    threshold = 0.70
    hot_pixels = np.argwhere(cam_rs > threshold)

    if len(hot_pixels) > 15: # Only draw if there's a distinct focal point
        y_min, x_min = hot_pixels.min(axis=0)
        y_max, x_max = hot_pixels.max(axis=0)

        # Padding around the hot cluster
        pad = 12
        x0 = max(x_min - pad, 0)
        y0 = max(y_min - pad, 0)
        x1 = min(x_max + pad, IMG_SIZE)
        y1 = min(y_max + pad, IMG_SIZE)

        # Sci-Fi / Medical Corner brackets instead of a clunky full square
        color = (0, 255, 160, 220)
        t = 12  # Length of corner legs
        w = 2   # Stroke width
        
        # Top-Left
        draw.line((x0, y0, x0+t, y0), fill=color, width=w)
        draw.line((x0, y0, x0, y0+t), fill=color, width=w)
        # Top-Right
        draw.line((x1, y0, x1-t, y0), fill=color, width=w)
        draw.line((x1, y0, x1, y0+t), fill=color, width=w)
        # Bottom-Left
        draw.line((x0, y1, x0+t, y1), fill=color, width=w)
        draw.line((x0, y1, x0, y1-t), fill=color, width=w)
        # Bottom-Right
        draw.line((x1, y1, x1-t, y1), fill=color, width=w)
        draw.line((x1, y1, x1, y1-t), fill=color, width=w)

    return overlay.convert('RGB')


# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(model, pil_img: Image.Image, device: torch.device):
    """Return (predicted_class_str, confidence_dict, class_idx)."""
    tensor = _preprocess(pil_img).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    class_names = CLASS_NAMES
    pred_idx    = int(np.argmax(probs))
    pred_class  = class_names[pred_idx]
    conf_dict   = {c: float(round(p * 100, 2)) for c, p in zip(class_names, probs)}

    return pred_class, conf_dict, pred_idx


# ── Full Pipeline ──────────────────────────────────────────────────────────────
def full_pipeline(model, image_bytes: bytes, device: torch.device) -> dict:
    """
    Entry point called by Flask.

    Returns a dict with:
        - original_b64   : base64 PNG of resized original
        - enhanced_b64   : base64 PNG of IFFT-enhanced image
        - overlay_b64    : base64 PNG with GradCAM heatmap (or plain enhanced if no tumor)
        - prediction     : class name string
        - confidences    : {class: pct}
        - has_tumor      : bool
    """
    pil_img = load_image(image_bytes)

    pred_class, conf_dict, pred_idx = run_inference(model, pil_img, device)
    has_tumor = pred_class in TUMOR_CLASSES

    enhanced_img = ifft_enhance(pil_img)

    if has_tumor:
        try:
            gradcam   = GradCAM(model)
            tensor    = _preprocess(pil_img).to(device)
            cam       = gradcam.generate(tensor, pred_idx)
            gradcam.remove_hooks()
            overlay_img = apply_heatmap_overlay(enhanced_img, cam)
        except Exception as e:
            print(f"[GradCAM warn] {e} — falling back to plain enhanced")
            overlay_img = enhanced_img
    else:
        overlay_img = enhanced_img   

    def to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.resize((IMG_SIZE, IMG_SIZE)).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    return {
        'original_b64' : to_b64(pil_img),
        'enhanced_b64' : to_b64(enhanced_img),
        'overlay_b64'  : to_b64(overlay_img),
        'prediction'   : pred_class,
        'confidences'  : conf_dict,
        'has_tumor'    : has_tumor,
    }