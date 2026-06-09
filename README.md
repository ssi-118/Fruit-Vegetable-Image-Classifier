---
title: Fruit Vegetable Classifier
emoji: 🍎
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---

# Fruit & Vegetable Image Classifier

A Flask web application that classifies fruit and vegetable images using a TensorFlow/Keras deep learning model.

## Dataset

The project uses an image dataset of fruits and vegetables arranged in class-wise folders.

Expected dataset structure:

```text
assets/
  train/
    apple/
    banana/
    carrot/
    ...
  validation/
    apple/
    banana/
    carrot/
    ...
```

Each folder name is treated as a class label.

## Algorithm

The model uses **Transfer Learning** with **MobileNetV2**.

- Base model: MobileNetV2 pretrained on ImageNet
- Custom layers: GlobalAveragePooling2D, Dense, Dropout, Softmax output
- Input image size: 224 x 224
- Loss function: Categorical Crossentropy
- Optimizer: Adam

## Best Algorithm

MobileNetV2 is used as the best algorithm for this project because it is lightweight, fast, and gives good accuracy for image classification tasks with limited training time.

## Basic Metrics

During training, the model tracks:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

The final prediction shows the top 3 classes with confidence scores.

## Frameworks and Libraries

- Python
- Flask
- TensorFlow / Keras
- NumPy
- Pillow
- HTML / CSS

## How to Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python assets/train_model.py
```

This creates:

```text
model/fruit_model.h5
model/labels.txt
```

3. Run the Flask app:

```bash
python app.py
```

4. Open in browser:

```text
http://127.0.0.1:5000
```

Upload a fruit or vegetable image and view the predicted class with confidence scores.
