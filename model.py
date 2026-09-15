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

# set device to gpu

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# get raw data + folders

raw_data = Path("data/cell_images")
class_folders = []
for folder in raw_data.iterdir():
    if folder.is_dir():
        class_folders.append(folder)

# create clean path

clean_data = Path("data/clean")
if clean_data.exists():
    shutil.rmtree(clean_data)
clean_data.mkdir(parents=True, exist_ok=True)

# clean the data

for folder in class_folders:
    class_name = folder.name
    clean_class_folder = clean_data / class_name
    clean_class_folder.mkdir(parents=True, exist_ok=True)
    images = list(folder.glob("*"))
    for image in images:
        try:
            with Image.open(image) as img:
                img.load()
                img = img.convert("RGB")
                save_path = clean_class_folder / f"{image.stem}.jpg"
                img.save(save_path, format="JPEG")
        except Exception:
            print("IMAGE SKIPPED")

# create split path + folders

split_data = Path("data/split")
if split_data.exists():
    shutil.rmtree(split_data)
train_data = split_data / "train"
val_data = split_data / "val"
test_data = split_data / "test"
train_data.mkdir(parents=True, exist_ok=True)
val_data.mkdir(parents=True, exist_ok=True)
test_data.mkdir(parents=True, exist_ok=True)

clean_class_folders = []
for folder in clean_data.iterdir():
    if folder.is_dir():
        clean_class_folders.append(folder)

# split data
random.seed(23)
for folder in clean_class_folders:
    class_name = folder.name
    images = list(folder.glob("*"))
    random.shuffle(images)
    split1 = int(len(images) * 0.70)
    split2 = int(len(images) * 0.85)
    train_images = images[:split1]
    val_images = images[split1:split2]
    test_images = images[split2:]
    train_folder = train_data / class_name
    val_folder = val_data / class_name
    test_folder = test_data / class_name
    train_folder.mkdir(parents=True, exist_ok=True)
    val_folder.mkdir(parents=True, exist_ok=True)
    test_folder.mkdir(parents=True, exist_ok=True)

    for image in train_images:
        shutil.copy(image, train_folder / image.name)
    for image in val_images:
        shutil.copy(image, val_folder / image.name)
    for image in test_images:
        shutil.copy(image, test_folder / image.name)

# create transformers and such

train_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]

    )
])

val_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]

    )
])

test_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]

    )
])

full_train_data = datasets.ImageFolder(
    root="data/split/train",
    transform=train_transformer
)

full_val_data = datasets.ImageFolder(
    root="data/split/val",
    transform=val_transformer
)

full_test_data = datasets.ImageFolder(
    root="data/split/test",
    transform=test_transformer
)

train_loader = DataLoader(
    full_train_data,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    full_val_data,
    batch_size=32,
    shuffle=False

)

test_loader = DataLoader(
    full_test_data,
    batch_size=32,
    shuffle=False

)

# load model

weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(

    weights=weights
)


# freeze parameters

for parameter in model.parameters():
    parameter.requires_grad = False

# change final

model.fc = nn.Linear(
    model.fc.in_features,
    2


)

model = model.to(device)

# create optimizer and loss

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.fc.parameters(),lr=0.001, weight_decay=0.0001)

# TRAIN AND VAL ONE

model.train()
best_accuracy = 0
times_since_last_upgrade = 0
for epoch in range(10):
    tracking_loss1 = 0
    train_correct = 0
    train_total = 0
    for image, label in train_loader:
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(image)
        prediction = torch.argmax(output, dim=1)
        train_correct += (prediction == label).sum().item()
        train_total += label.size(0)
        loss = loss_func(output, label)
        tracking_loss1 += loss.item()
        loss.backward()
        optimizer.step()

    train_loss = tracking_loss1 / len(train_loader)
    train_accuracy = train_correct / train_total


    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    with torch.no_grad():
        for image,label in val_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss = loss_func(output, label)
            val_loss += loss.item()
            prediction = torch.argmax(output, dim=1)
            val_correct += (prediction == label).sum().item()
            val_total += label.size(0)

    val_loss = val_loss / len(train_loader)
    val_accuracy = val_correct / val_total


    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "bestmodel.pth"

        )

        times_since_last_upgrade = 0

    else:

        times_since_last_upgrade += 1


    if times_since_last_upgrade > 4:
        break


    print(f"EPOCH {epoch + 1}")
    print(f"TRAIN LOSS: {train_loss}")
    print(f"TRAIN ACCURACY: {train_accuracy}")
    print(f"VAL LOSS: {val_loss}")
    print(f"VAL ACCURACY: {val_accuracy}")


# fine tune

for parameter in model.layer4.parameters():
    parameter.requires_grad = True


trainable_parameters = []
for parameter in model.parameters():
    if parameter.requires_grad == True:
        trainable_parameters.append(parameter)

optimizer2 = torch.optim.AdamW(trainable_parameters,lr=0.0001, weight_decay=0.0001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer2,
    mode="min",
    patience=4,
    factor=0.5


)

# TRAIN VAL TWO
times_since_last_upgrade = 0
model.train()
for epoch in range(5):
    train_correct2 = 0
    train_total2 = 0
    tracking_loss2 = 0
    model.train()
    for image, label in train_loader:
        image = image.to(device)
        label = label.to(device)
        optimizer2.zero_grad()
        output = model(image)
        prediction = torch.argmax(output, dim=1)
        train_correct2 += (prediction == label).sum().item()
        train_total2 += label.size(0)
        loss = loss_func(output, label)
        tracking_loss2 += loss.item()
        loss.backward()
        optimizer2.step()

    train_loss2 = tracking_loss2 / len(train_loader)
    train_accuracy2 = train_correct2 / train_total2


    model.eval()
    val_correct2 = 0
    val_total2 = 0
    val_loss2 = 0
    with torch.no_grad():
        for image, label in val_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            prediction = torch.argmax(output, dim=1)
            val_correct2 += (prediction == label).sum().item()
            val_total2 += label.size(0)
            loss = loss_func(output, label)
            val_loss2 += loss.item()





    val_loss2 = val_loss2 / len(val_loader)
    val_accuracy2 = val_correct2 / val_total2

    scheduler.step(val_loss2)

    if val_accuracy2 > best_accuracy:
        best_accuracy = val_accuracy2

        torch.save(
            model.state_dict(),
            "bestmodel.pth"
        )

        times_since_last_upgrade = 0

    else:
        times_since_last_upgrade += 1


    if times_since_last_upgrade == 4:
        break


    print(f"EPOCH: {epoch + 1}")
    print(f"TRAIN LOSS: {train_loss2}")
    print(f"TRAIN ACCURACY: {train_accuracy2}")
    print(f"VAL LOSS: {val_loss2}")
    print(f"VAL ACCURACY {val_accuracy2}")


# Load Best Model

model.load_state_dict(


    torch.load("bestmodel.pth")

)

model.eval()
correct = 0
total = 0
all_predictions = []
all_labels = []
with torch.no_grad():
    for image, label in test_loader:
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        prediction = torch.argmax(output,dim=1)
        correct += (prediction == label).sum().item()
        total += label.size(0)
        all_predictions.extend(prediction.tolist())
        all_labels.extend(label.tolist())

print(f"FINAL ACCURACY: {correct / total}")

confusion = confusion_matrix(all_labels, all_predictions)
print(f"CONFUSION: {confusion}")