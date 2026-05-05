# src/predict.py

import argparse
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = "models/resnet18_fc_only_best.pth"
CLASS_NAMES_PATH = "models/class_names.json"
IMAGE_SIZE = 224
TOP_K = 5


def load_model(device):
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    num_classes = len(class_names)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    return model, class_names


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, class_names = load_model(device)

    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=TOP_K, dim=1)

    print(f"\nImage: {image_path}")
    print(f"Top-{TOP_K} predictions:")
    for rank, (prob, index) in enumerate(zip(top_probs[0], top_indices[0]), start=1):
        class_name = class_names[index.item()]
        confidence = prob.item() * 100
        print(f"{rank}. {class_name} ({confidence:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    predict(args.image)