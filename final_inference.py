import io
import json
import os
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# Keep consistent with training label order
CLASS_LABELS = ["1", "2", "3", "4", "5"]
MODEL_FILENAME = "model.pth"   # update if your artifact uses a different filename

_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def model_fn(model_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_LABELS))

    model_path = os.path.join(model_dir, MODEL_FILENAME)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    # We expect raw image bytes
    if request_content_type in ("application/x-image", "application/octet-stream"):
        img = Image.open(io.BytesIO(request_body)).convert("RGB")
        x = _transform(img).unsqueeze(0)
        return x
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    device = next(model.parameters()).device
    x = input_data.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    conf, idx = torch.max(probs, dim=0)
    idx = int(idx.item())
    return {
        "predicted_label": CLASS_LABELS[idx],
        "predicted_index": idx,
        "confidence": float(conf.item()),
        "probabilities": [float(p) for p in probs.cpu().tolist()],
        "class_labels": CLASS_LABELS
    }

def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"