from calendar import day_abbr
from enum import show_flag_values
from idlelib.run import eof
from os import mkdir
from os.path import exists, split
from pathlib import Path
import random
import shutil
from types import coroutine
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder, folder
from PIL import Image

# rebuild structure

model = models.resnet50(

    weights=None
)

model.fc = nn.Linear(

    model.fc.in_features,
    2


)

# load model

model.load_state_dict(
    torch.load("bestmodel.pth")


)

# prediction

model.eval()


predicted_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]

    )
])

class_names = [

    "Parasitized",
    "Uninfected"


]

image_path = Path("cell2.jpg")

with Image.open(image_path) as img:
    img = img.convert("RGB")
    original_image = img.copy()
    image = predicted_transformer(img)

image = image.unsqueeze(0)

with torch.no_grad():
    output = model(image)
    prediction = torch.argmax(output,dim=1)
    probabilities = torch.softmax(output, dim=1)
    confidence = torch.max(probabilities).item()

class_prediction = class_names[prediction]
print(class_prediction)

# globals

activations = None
gradients = None

# Functions

def save_activations(module, input, output):
    global activations
    activations = output

def save_gradients(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# target

target_layer = model.layer4[-1].conv3

# Hooks

forward_hook = target_layer.register_forward_hook(
    save_activations
)

backward_hook = target_layer.register_full_backward_hook(
    save_gradients
)

# Model Stuff

model.zero_grad()
output = model(image)
class_score = output[0,prediction]
class_score.backward()

# weights

weights = gradients.mean(
    dim=(2,3),
    keepdim=True
)

weighted_activations = weights * activations

# heatmap

heatmap = weighted_activations.sum(
    dim=1
)

heatmap = torch.relu(heatmap)
heatmap = heatmap.squeeze()
heatmap = heatmap.detach().cpu().numpy()

# Normalize

heatmap = heatmap - heatmap.min()

if heatmap.max() > 0:
    heatmap = heatmap / heatmap.max()

# image

heatmap_image = Image.fromarray(
    np.uint8(heatmap * 255)
)

heatmap_image = heatmap_image.resize(
    original_image.size
)

heatmap = np.array(heatmap_image) / 255

# Show

plt.imshow(
    original_image
)

plt.imshow(
    heatmap,
    cmap="jet",
    alpha=0.45
)

plt.axis("off")

plt.title(f"{class_prediction} : {confidence * 100}")
plt.show()