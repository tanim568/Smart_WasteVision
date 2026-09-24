"""Smart WasteVision — Gradio inference app (Render Free compatible).

Loads best_model_resnet18_weighted_finetuned.pth, which was produced by the training
notebook (Section 11). The architecture below must match the notebook's
build_resnet18_classifier(): ResNet18 with fc = Dropout(0.4) + Linear(512, 6).
"""
import os

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_resnet18_weighted_finetuned.pth")

# Same order as the notebook's CLASS_TO_IDX (alphabetical).
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Render Free has no GPU and very little CPU/RAM.
DEVICE = torch.device("cpu")
torch.set_num_threads(1)

# Same preprocessing as the notebook's validation/test transform.
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {MODEL_PATH}. Commit "
            "best_model_resnet18_weighted_finetuned.pth next to app.py."
        )

    model = models.resnet18(weights=None)          # no pretrained download; weights come from the checkpoint
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, len(CLASSES)),
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)              # strict: fails loudly on any mismatch
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()


def predict(image):
    if image is None:
        return {}

    tensor = inference_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    return {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a waste image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="Smart WasteVision",
    description=(
        "Classifies a waste image as cardboard, glass, metal, paper, plastic or trash "
        "using a fine-tuned ResNet18 trained on TrashNet."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    # Render provides the port in $PORT and requires binding to 0.0.0.0.
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
