import io
import os
import json
import base64
import logging
from typing import Any, Dict

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Class labels must match ImageFolder order used in training: ['1','2','3','4','5']
CLASS_LABELS = ["1", "2", "3", "4", "5"]

def build_model(num_classes: int = 5) -> nn.Module:
    model = models.resnet50(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model

# Preprocessing must match eval transforms from training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

EVAL_TFMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def model_fn(model_dir: str) -> nn.Module:
    """
    Load model from the SageMaker model directory.
    SageMaker extracts model.tar.gz into model_dir.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(CLASS_LABELS))

    model_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pth not found in: {model_dir}. Files: {os.listdir(model_dir)}")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    logger.info(f"✅ Loaded model from {model_path} on device={device}")
    return model

def _bytes_to_tensor(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = EVAL_TFMS(img).unsqueeze(0)  # shape: [1,3,224,224]
    return x

def input_fn(request_body: bytes, content_type: str) -> torch.Tensor:
    """
    Convert incoming payload to a torch.Tensor.
    Supports raw image bytes or JSON containing base64.
    """
    if content_type in ("application/x-image", "application/octet-stream", "image/jpeg", "image/jpg", "image/png"):
        return _bytes_to_tensor(request_body)

    if content_type == "application/json":
        payload = json.loads(request_body.decode("utf-8"))
        # expected JSON formats:
        # {"image_b64": "<base64 string>"}
        # {"b64": "<base64 string>"}
        b64 = payload.get("image_b64") or payload.get("b64")
        if not b64:
            raise ValueError("JSON input must include 'image_b64' or 'b64' field.")
        image_bytes = base64.b64decode(b64)
        return _bytes_to_tensor(image_bytes)

    raise ValueError(f"Unsupported content_type: {content_type}")

def predict_fn(input_data: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
    """
    Run inference and return a JSON-serializable dict.
    """
    device = next(model.parameters()).device
    x = input_data.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # [C]

    top_prob, top_idx = torch.max(probs, dim=0)
    pred_label = CLASS_LABELS[int(top_idx)]

    return {
        "predicted_label": pred_label,
        "predicted_index": int(top_idx),
        "confidence": float(top_prob),
        "probabilities": [float(p) for p in probs.cpu().numpy().tolist()],
        "class_labels": CLASS_LABELS,
    }

def output_fn(prediction: Dict[str, Any], accept: str) -> bytes:
    """
    Serialize prediction to JSON.
    """
    if accept in ("application/json", "*/*"):
        return json.dumps(prediction).encode("utf-8")
    raise ValueError(f"Unsupported accept type: {accept}")