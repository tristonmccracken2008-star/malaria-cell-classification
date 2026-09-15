# Malaria Cell Classification

A deep learning project that classifies microscope images of red blood cells as either **Parasitized** or **Uninfected** using transfer learning with ResNet50.

I built this project to practice creating a complete image classification pipeline on a real-world dataset. I started with the raw images and worked through cleaning the data, creating train/validation/test splits, data augmentation, transfer learning, fine-tuning, evaluation, and finally making predictions on new images.

## Results

My final fine-tuned ResNet50 achieved **96.59% accuracy** on the held-out test set.

- Test images: **4,134**
- Correct predictions: **3,993**
- Incorrect predictions: **141**
- Final test accuracy: **96.59%**

Confusion matrix:

```text
[[1980   87]
 [  54 2013]]
```

The best validation accuracy was also around **96.6%**, which was very close to the final test result.

## Dataset

I used the **NIH/NLM Malaria Cell Images Dataset**, which contains microscope images of individual red blood cells.

There are two classes:

- **Parasitized** — a malaria parasite is present in the cell
- **Uninfected** — no malaria parasite is visible in the cell

Dataset source:

https://data.lhncbc.nlm.nih.gov/public/Malaria/

Before training the model, I wrote a cleaning step that attempted to open every image with PIL. If an image could not be opened, it was skipped. The valid images were placed into a new clean dataset.

I then split the cleaned data into:

- **70% training**
- **15% validation**
- **15% testing**

I kept the test set separate so that it could be used for the final evaluation instead of influencing training or model selection.

## Data Augmentation and Preprocessing

For the training data, I used several image transformations to give the model more variation during training.

These included:

- Resizing
- Random horizontal flips
- Random rotations
- Conversion to PyTorch tensors
- ImageNet normalization

I used a separate transformation pipeline for validation and testing without the random augmentations.

The images were then loaded using PyTorch **DataLoaders**, which allowed the model to train using mini-batches and shuffle the training data between epochs.

## Model

For the model, I used a **ResNet50 convolutional neural network pretrained on ImageNet**.

Instead of training the entire network from scratch, I used **transfer learning**. This allowed me to take advantage of visual features ResNet50 had already learned and adapt them to the malaria classification problem.

I replaced ResNet50's original classification layer with a new fully connected layer that outputs the two classes in this project.

## Training

I trained the model in two main stages.

### Stage 1: Feature Extraction

First, I froze the pretrained ResNet50 layers and trained the new classification layer.

This let the new classifier learn the difference between Parasitized and Uninfected cells while keeping the pretrained feature extractor unchanged.

I trained and validated the model for **10 epochs** during this stage.

### Stage 2: Fine-Tuning

After the first training stage, I unfroze deeper layers of ResNet50 and continued training for another **5 epochs** with a lower learning rate.

This allowed some of the pretrained features to adjust more specifically to the malaria cell images without making overly large changes to the network.

Fine-tuning increased validation accuracy from around **94% to 96.6%**.

## Training Setup

The main training setup included:

- **PyTorch**
- **Torchvision**
- **ResNet50**
- **Transfer learning**
- **CrossEntropyLoss**
- **AdamW optimizer**
- **Weight decay**
- **Learning rate scheduling**
- **Data augmentation**
- **ImageNet normalization**
- **Model checkpointing**
- **Apple MPS acceleration**

During every epoch, I tracked both training and validation loss and accuracy.

I also saved the model whenever it reached a new best validation accuracy. This became especially useful during fine-tuning because training accuracy continued increasing even after validation performance started to level off.

Instead of assuming the final epoch was the best model, I reloaded the checkpoint with the best validation performance before evaluating it on the test set.

## Evaluation

After training and fine-tuning were complete, I evaluated the best model on the held-out test set.

The final accuracy was:

**96.59%**

I also collected all of the predictions and true labels and created a **confusion matrix** to see where the model was making mistakes instead of only looking at overall accuracy.

```text
[[1980   87]
 [  54 2013]]
```

This gave me a better picture of how the model performed on both classes.

## Predicting New Images

I created a separate `predict.py` script so I could use the trained model on individual images outside of the training loop.

The prediction pipeline:

1. Recreates the ResNet50 architecture
2. Loads the trained model weights
3. Opens an image and converts it to RGB
4. Applies the same preprocessing used for evaluation
5. Adds a batch dimension
6. Runs the image through the model with gradients disabled
7. Finds the class with the highest model output
8. Returns **Parasitized** or **Uninfected**

I also tried the model on individual cells cropped from microscope images outside of the original training workflow as a basic real-world sanity check.

## Project Structure

```text
malaria-cell-classification/
│
├── model.py       # Data preparation, training, fine-tuning, and evaluation
├── predict.py     # Predictions on new cell images
├── .gitignore
└── README.md
```

The dataset and trained `.pth` model weights are not included in the GitHub repository.

## What I Learned

This project helped me put together a lot of the deep learning concepts I had been learning separately into one complete project.

Some of the main concepts I worked with were:

- Convolutional neural networks
- Transfer learning
- ResNet architectures
- Feature extraction
- Fine-tuning
- Freezing and unfreezing model layers
- Data augmentation
- ImageNet normalization
- Mini-batch training with DataLoaders
- Train/validation/test splitting
- Avoiding data leakage
- Cross-entropy loss
- AdamW optimization
- Learning rates
- Weight decay
- Learning rate scheduling
- Detecting overfitting using validation performance
- Model checkpointing
- Confusion matrices
- Inference on new images
- Apple MPS hardware acceleration

One of the most useful things I learned was the difference between simply getting a model to train and actually building a training process where I could use validation performance to make decisions and then evaluate the final model on data it had not been trained on.

## Limitations

The biggest limitation of the current model is that it expects an image of an **already-cropped individual red blood cell**.

It cannot take a full microscope image and automatically find the cells. That would require an object detection step before classification.

The current dataset is also split at the image level. For a more rigorous medical imaging experiment, I would want to investigate splitting the data by **patient** so images from the same patient could not appear across training and evaluation sets.

This is an educational machine learning project and is **not intended to be used for medical diagnosis or clinical decision-making**.

## Future Work

There are several things I want to add as I continue improving the project:

- Add prediction probabilities/confidence scores
- Use **Grad-CAM** to visualize which parts of a cell influence the model's predictions
- Analyze misclassified cells in more detail
- Compare ResNet50 with other architectures
- Experiment with different augmentation and fine-tuning strategies
- Test patient-level dataset splitting
- Build a model that detects individual cells in full microscope images
- Send those detected cells through the current ResNet50 classifier
- Eventually combine detection and classification into one larger pipeline

The current system works like this:

```text
Individual Cell Image
        ↓
ResNet50
        ↓
Parasitized / Uninfected
```

A future version could work more like:

```text
Full Microscope Image
        ↓
Cell Detection Model
        ↓
Detected Cell Crops
        ↓
ResNet50 Classifier
        ↓
Parasitized / Uninfected
        ↓
Overall Image Analysis
```

That would take the project beyond classification of prepared cell images and toward analyzing complete microscope images.

## Tech Stack

**Python · PyTorch · Torchvision · ResNet50 · PIL/Pillow · scikit-learn · Apple MPS**