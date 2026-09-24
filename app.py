import torch
import gradio as gr
from PIL import Image
from torchvision import models, transforms

classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, len(classes))
)

model.load_state_dict(
    torch.load(
        "best_model_resnet18_weighted_finetuned.pth",
        map_location=device
    )
)
model.to(device)
model.eval()

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(image: Image.Image):
    image = image.convert("RGB")
    tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, 3)

    return {
        classes[idx.item()]: float(prob)
        for prob, idx in zip(top_probs, top_indices)
    }

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Smart WasteVision",
    description="AI-powered waste classification using Fine-Tuned ResNet18."
)

if __name__ == "__main__":
    demo.launch(share=True)
