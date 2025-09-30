# main.py
import json
import toml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import (
    ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
)

# -------------------------
# Configurations
# -------------------------
with open("config.json") as f:
    config = json.load(f)

params = toml.load("params.toml")

with open("grid_search.json") as f:
    grid = json.load(f)

# -------------------------
# Device & mixed precision
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"

# -------------------------
# Run mode: demo or full
# -------------------------
FULL_RUN = False  # set True for full CIFAR-10 training

# -------------------------
# Data loaders
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

if FULL_RUN:
    train_data = train_dataset
    val_data = val_dataset
else:
    train_data = Subset(train_dataset, range(500))
    val_data = Subset(val_dataset, range(100))

train_loader = DataLoader(
    train_data, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=config["data"]["num_workers"]
)
val_loader = DataLoader(
    val_data, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"]
)

# -------------------------
# Model helper
# -------------------------
model_dict = {
    "resnet34": (models.resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V1),
    "resnet152": (models.resnet152, ResNet152_Weights.IMAGENET1K_V1)
}

def get_model(name):
    model_fn, weights_enum = model_dict[name]
    model = model_fn(weights=weights_enum if config["models"][name]["pretrained"] else None)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10
    return model

# -------------------------
# Training function
# -------------------------
def train_model(model, optimizer_name, lr, momentum, epochs=1, train_loader=train_loader, val_loader=val_loader):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Ensure optimizer sees only trainable parameters
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found! Check requires_grad settings.")

    optimizer = (
        optim.Adam(trainable_params, lr=lr) if optimizer_name.lower() == "adam"
        else optim.SGD(trainable_params, lr=lr, momentum=momentum)
    )

    scaler = torch.amp.GradScaler(enabled=use_amp)

    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    # Validation first batch inference
    model.eval()
    preds_list = []
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds_list.extend(outputs.argmax(dim=1).cpu().tolist())
            break
    return preds_list

# -------------------------
# Evaluate function
# -------------------------
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# -------------------------
# a. Model inference
# -------------------------
print("\n--- a. Model Inference Outputs ---")
for model_name in config["models"]:
    model = get_model(model_name)
    hp = params[model_name]
    preds = train_model(model, hp["optimizer"], hp["learning_rate"], hp["momentum"])
    print(f"{model_name} Predictions: {preds}")

# -------------------------
# b. JSON config display
# -------------------------
print("\n--- b. Data & Model JSON ---")
print(json.dumps(config, indent=4))

# -------------------------
# c. TOML hyperparameters display
# -------------------------
print("\n--- c. Model Hyperparameters TOML ---")
for model_name, hp in params.items():
    print(f"[{model_name}] learning_rate={hp['learning_rate']}, optimizer={hp['optimizer']}, momentum={hp['momentum']}")

# -------------------------
# d. Integrated pipeline
# -------------------------
print("\n--- d. Integrated Pipeline ---")
for model_name in config["models"]:
    hp = params[model_name]
    print(f"Training {model_name} with lr={hp['learning_rate']}, optimizer={hp['optimizer']}, momentum={hp['momentum']}")

# -------------------------
# e. Robust Grid Search for ResNet34
# -------------------------
print("\n--- e. Grid search for ResNet34 ---")
best_accuracy = 0
best_params = {}

for lr in grid["learning_rate"]:
    for opt in grid["optimizer"]:
        for mom in grid["momentum"]:
            model = get_model("resnet34")
            model.to(device)

            # Freeze backbone, train only final layer
            for name, param in model.named_parameters():
                param.requires_grad = name.startswith("fc")

            # Train model
            train_model(model, optimizer_name=opt, lr=lr, momentum=mom, epochs=1)

            # Evaluate validation accuracy
            accuracy = evaluate_model(model, val_loader)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"learning_rate": lr, "optimizer": opt, "momentum": mom}

print(f"Best hyperparameters for resnet34: {best_params}, Accuracy: {best_accuracy:.4f}")