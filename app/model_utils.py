"""
model_utils.py
──────────────
Handles loading .pth model files saved from the training notebook.
Supports the save format:
    {
        'model_name'  : 'efficientnet_b0',
        'num_classes' : 4,
        'class_names' : [...],
        'state_dict'  : ...,
        'best_val_acc': float,
        'img_size'    : 224
    }
"""

import os
import torch
import timm

DEVICE = torch.device('cpu')   # Flask app always runs on CPU (potato-PC safe)


def list_models(models_dir: str) -> list[str]:
    """Return sorted list of .pth files in the models directory."""
    if not os.path.isdir(models_dir):
        return []
    return sorted([f for f in os.listdir(models_dir) if f.endswith('.pth')])


def load_model(model_path: str) -> tuple:
    """
    Load a saved model checkpoint.

    Returns:
        (model, meta_dict)
        model     – PyTorch model in eval mode on CPU
        meta_dict – {'class_names', 'num_classes', 'img_size', 'best_val_acc'}
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # ── Support both full checkpoint dict and raw state_dict ──
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_name  = checkpoint.get('model_name',   'efficientnet_b0')
        num_classes = checkpoint.get('num_classes',  4)
        class_names = checkpoint.get('class_names',  ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'])
        img_size    = checkpoint.get('img_size',      224)
        best_acc    = checkpoint.get('best_val_acc',  None)
        state_dict  = checkpoint['state_dict']
    else:
        # Raw state_dict fallback
        model_name  = 'efficientnet_b0'
        num_classes = 4
        class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        img_size    = 224
        best_acc    = None
        state_dict  = checkpoint

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    meta = {
        'class_names' : class_names,
        'num_classes' : num_classes,
        'img_size'    : img_size,
        'best_val_acc': best_acc,
        'model_name'  : model_name,
    }

    return model, meta