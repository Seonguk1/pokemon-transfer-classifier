import os
import json
import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


DATA_DIR = "data/pokemon"
MODEL_DIR = "models"
RESULT_DIR = "results"

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = 224


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        "val": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        "test": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    }

    image_datasets = {
        x: datasets.ImageFolder(
            root=os.path.join(DATA_DIR, x),
            transform=data_transforms[x]
        )
        for x in ["train", "val", "test"]
    }

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2
        ),
        "test": DataLoader(
            image_datasets["test"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )
    }

    class_names = image_datasets["train"].classes
    num_classes = len(class_names)

    print(f"Number of classes: {num_classes}")
    print(f"Classes example: {class_names[:10]}")

    with open(os.path.join(MODEL_DIR, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 30)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = accuracy_score(all_labels, all_preds)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "train":
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_model_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_weights)

    model_path = os.path.join(MODEL_DIR, "resnet18_fc_only_best.pth")
    torch.save(model.state_dict(), model_path)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Saved model: {model_path}")

    save_learning_curve(train_losses, val_losses, train_accs, val_accs)

    evaluate_model(model, dataloaders["test"], image_datasets["test"], device)


def save_learning_curve(train_losses, val_losses, train_accs, val_accs):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet18 Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, "resnet18_loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ResNet18 Accuracy Curve")
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, "resnet18_accuracy_curve.png"))
    plt.close()

    print("Saved learning curves in results folder.")


def evaluate_model(model, test_loader, test_dataset, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="test"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print("\nTest Results")
    print("-" * 30)
    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall   : {recall:.4f}")
    print(f"Test F1-score : {f1:.4f}")

    result = {
        "experiment": "ResNet18 pretrained - FC only",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    with open(os.path.join(RESULT_DIR, "resnet18_fc_only_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Saved test result: results/resnet18_fc_only_result.json")


if __name__ == "__main__":
    main()