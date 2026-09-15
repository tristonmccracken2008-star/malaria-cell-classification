from calendar import day_abbr
from enum import show_flag_values
from idlelib.run import eof
from os import mkdir
from os.path import exists, split
from pathlib import Path
import random
import shutil
from types import coroutine
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
    torch.load("ENTER PATH HERE")


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

image_path = Path("2710_lores.jpg")

with Image.open(image_path) as img:
    img = img.convert("RGB")
    image = predicted_transformer(img)

image = image.unsqueeze(0)

with torch.no_grad():
    output = model(image)
    prediction = torch.argmax(output,dim=1)

class_prediction = class_names[prediction]
print(class_prediction)
