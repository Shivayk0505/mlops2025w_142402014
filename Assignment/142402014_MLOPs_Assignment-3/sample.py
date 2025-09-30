import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import toml
import itertools
from torchvision import datasets, transforms

#-----------------------------------------
#  Load Dataset
#----------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = datasets.CIFAR10(root="./data/", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root="./data/", train=False, download=True, transform=transform)


# -------------------------
# Load pretrained ResNet
# -------------------------
def get_resnet(model_name, num_classes=1000):
    if model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif model_name == "resnet152":
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    else:
        raise ValueError("Invalid ResNet model")
    
    # Modify last layer for classification (example: 10 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# -------------------------
# Load JSON & TOML configs
# -------------------------
with open("config.json") as f:
    config = json.load(f)

params = toml.load("params.toml")

# Example: Pick model and parameters
for model_cfg in config["models"]:
    model_name = model_cfg["name"]
    model = get_resnet(model_name, num_classes=10)

    # Get hyperparams
    hp = params[model_name]
    lr = hp["learning_rate"]
    batch_size = hp["batch_size"]
    epochs = hp["epochs"]
    optimizer_type = hp["optimizer"]

    # Define optimizer
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print(f"Training {model_name} with lr={lr}, optimizer={optimizer_type}")


# -------------------------
# Grid Search
# -------------------------
with open("grid.json") as f:
    grid = json.load(f)

grid_params = grid["hyperparameters"]

def grid_search(model_name):
    search_space = list(itertools.product(
        grid_params["learning_rate"],
        grid_params["optimizer"],
        grid_params["momentum"]
    ))
    
    for lr, opt, mom in search_space:
        model = get_resnet(model_name, num_classes=10)
        if opt == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)
        
        print(f"Grid Search -> {model_name}: lr={lr}, optimizer={opt}, momentum={mom}")

# Run grid search on ResNet34
grid_search("resnet34")

