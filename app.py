# app.py

import json
from pathlib import Path

import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import models, transforms


MODEL_PATH = Path("models/resnet18_full_best.pth")
CLASS_NAMES_PATH = Path("models/class_names.json")
IMAGE_SIZE = 224
TOP_K = 5


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    num_classes = len(class_names)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    return model, class_names, device


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor


def predict(image, model, class_names, device):
    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=TOP_K, dim=1)

    results = []

    for prob, index in zip(top_probs[0], top_indices[0]):
        class_name = class_names[index.item()]
        confidence = prob.item() * 100
        results.append((class_name, confidence))

    return results


def main():
    st.set_page_config(
        page_title="Pokemon Transfer Classifier",
        page_icon="⚡",
        layout="centered"
    )

    st.title("Pokemon Transfer Classifier")
    st.write("Upload a Pokemon image and the model will predict the Pokemon name.")

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    if not CLASS_NAMES_PATH.exists():
        st.error(f"Class names file not found: {CLASS_NAMES_PATH}")
        st.stop()

    model, class_names, device = load_model()

    st.info(f"Device: {device}")
    st.write(f"Number of classes: {len(class_names)}")

    uploaded_file = st.file_uploader(
        "Upload Pokemon image",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.subheader("Input Image")
        st.image(image, use_container_width=True)

        results = predict(image, model, class_names, device)

        st.subheader("Top-5 Predictions")

        for rank, (class_name, confidence) in enumerate(results, start=1):
            st.write(f"{rank}. **{class_name}** - {confidence:.2f}%")
            st.progress(confidence / 100)


if __name__ == "__main__":
    main()