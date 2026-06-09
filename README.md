---
title: Fruit Vegetable Classifier
emoji: 🍎
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---

Here is a tailored, complete `README.md` for your **Fruit & Vegetable Image Classifier** project, structured precisely like your sample and fully optimized for a Hugging Face Spaces Docker deployment.

---

# Fruit & Vegetable Image Classifier

A web-based computer vision application powered by a deep learning model trained on image datasets of fresh produce. Upload an image of a fruit or vegetable, and the system instantly recognizes and predicts its specific category.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.3-lightgrey)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Dataset Used

This project utilizes a comprehensive dataset consisting of high-quality images of various fruits and vegetables to ensure robust categorization across different lighting conditions and angles.

- **Dataset:** Fruits and Vegetables Image Recognition Dataset
- **Total Images:** ~3,000+ images for training, validation, and testing
- **Categories:** Multiple distinct classes including apples, bananas, broccoli, carrots, tomatoes, and more.
- **Image Format:** JPG / PNG

## Supported Categories

Apples, Bananas, Beetroot, Bell Peppers, Cabbage, Capsicum, Carrots, Cauliflower, Chili Peppers, Corn, Cucumbers, Eggplants, Garlic, Ginger, Grapes, Jalapenos, Kiwi, Lemons, Lettuce, Mangoes, Onions, Oranges, Paprika, Pears, Peas, Pineapples, Pomegranates, Potatoes, Radishes, Soy Beans, Spinach, Sweet Potatoes, Tomatoes, Turnips, Watermelons.

---

## Features

- **Instant Recognition:** Upload image files (JPEG, PNG) for rapid inference.
- **Deep Learning Backend:** Powered by a Convolutional Neural Network (CNN) tuned for image classification tasks.
- **Robust Image Processing:** Automated resizing, normalization, and preparation pipeline for accurate processing.
- **User-Friendly Interface:** Clean web UI designed for straightforward image uploading and clear classification outputs.

---

## Project Structure


```

Fruit-Vegetable-Image-Classifier/
├── app.py                  # Flask backend server & inference routing
├── requirements.txt        # Python external dependencies
├── Dockerfile              # Docker container configuration for HF deployment
├── README.md               # Project documentation
├── model/
│   ├── fruit_veg_model.h5  # Trained TensorFlow/Keras deep learning model
│   └── labels.txt
├── static/
│   └── uploads/
├── docs/
│   └── Fruit_IP_OP.docx
├── assets/
│   └── train_model.py      # CNN Model architecture and training script
└── templates/
    ├── index.html
    └── result.html

```

---

## Running Locally

### 1. Clone the repository

```bash
git clone [https://github.com/ssi-118/Fruit-Vegetable-Image-Classifier.git](https://github.com/ssi-118/Fruit-Vegetable-Image-Classifier.git)
cd Fruit-Vegetable-Image-Classifier

```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

```

### 3. Install dependencies

```bash
pip install -r requirements.txt

```

### 4. Train the model (Optional)

If you wish to re-train the model with your custom dataset, organize your data into `train`, `validation`, and `test` directories and run:

```bash
python train_model.py

```

*If skipped, ensure your pre-trained `.h5` model file is saved inside the `model/` directory.*

### 5. Start the development server

```bash
python app.py

```

Visit **http://127.0.0.1:5000** in your web browser.

---

## Running with Docker

You can containerize the application locally to match the exact runtime behavior of the cloud deployment environment.

```bash
docker build -t fruit-veg-classifier .
docker run -p 7860:7860 fruit-veg-classifier

```

Visit **http://localhost:7860**

---

## Deployment

### Hugging Face Spaces (Docker SDK)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Select **Docker** as your SDK framework choice.
3. Ensure your `README.md` includes the exact configuration metadata block (YAML frontmatter) found at the top of this file.
4. Push your repository to your remote Hugging Face Space:

```bash
git init
git remote add origin [https://huggingface.co/spaces/your-username/Fruit-Vegetable-Image-Classifier](https://huggingface.co/spaces/your-username/Fruit-Vegetable-Image-Classifier)
git add .
git commit -m "feat: setup project for huggingface spaces"
git push origin main

```

The application will build automatically and display live at `https://your-username-Fruit-Vegetable-Image-Classifier.hf.space`

> ⚠️ **Important deployment note:** Hugging Face Spaces defaults to container port routing over port `7860`. Ensure your internal production server script (e.g., Gunicorn/Flask) binds to `0.0.0.0:7860` in your `Dockerfile`.

---

## 📦 Core Dependencies

| Package | Version | Purpose |
| --- | --- | --- |
| Flask | 3.0.3 | Web Framework application logic |
| gunicorn | 22.0.0 | Production WSGI HTTP server |
| tensorflow | 2.15.0 | Deep Learning compilation and inference runtime |
| Pillow | 10.3.0 | Image processing and manipulation backend |
| numpy | 1.26.4 | Multi-dimensional matrix calculations |

---

## 🧠 Model Architecture Details

| Parameter / Layer Property | Configuration Value |
| --- | --- |
| Base Architecture | Sequential Convolutional Neural Network (CNN) |
| Input Resolution | 224 x 224 pixels (RGB) |
| Optimization Algorithm | Adam Optimizer |
| Loss Function | Categorical Crossentropy |
| Target Classes | 36 distinct fruits and vegetables |

---

## Live Demo

🌐 https://soha118-fruit-vegetable-image-classifier.hf.space

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
