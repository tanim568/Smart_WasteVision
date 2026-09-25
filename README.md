# Smart WasteVision

An image classification application that identifies waste materials across six TrashNet categories using a fine-tuned ResNet18 model and a Gradio web interface.

## 🚀 Live Demo

**Try the deployed application:**  
https://smart-wastevision.onrender.com

Upload a waste image and the application returns the predicted class with the top-3 prediction probabilities.

## 📌 Project Overview

Smart WasteVision is a computer vision project built to classify waste images into six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The project uses **transfer learning with ResNet18**, with the final classification layer adapted for six classes. The model was fine-tuned on a cleaned subset of the TrashNet dataset and deployed as a lightweight Gradio application.

## ✨ Key Features

- 🗑️ Six-class waste image classification
- 🧠 ResNet18 transfer learning and fine-tuning
- ⚖️ Weighted `CrossEntropyLoss` for class imbalance
- 📊 Top-3 prediction probabilities
- 🖥️ Simple Gradio web interface
- ⚡ CPU-compatible inference
- 🌐 Deployed on Render

## 🧠 Model

The final model uses:

- **Architecture:** ResNet18
- **Input size:** 224 × 224
- **Backbone:** Pretrained ResNet18
- **Fine-tuned layers:** `layer4` and `fc`
- **Classifier:** Dropout(0.4) → Linear(512, 6)
- **Optimizer:** Adam
- **Learning rate:** 1e-4
- **Loss:** Weighted Cross Entropy
- **Training epochs:** 8

### Inference Pipeline

```text
Input Image
    ↓
Resize to 224×224
    ↓
ImageNet Normalization
    ↓
Fine-tuned ResNet18
    ↓
6-Class Prediction
    ↓
Top-3 Probabilities
```

## 📊 Dataset

The project uses a cleaned version of the **TrashNet** dataset.

| Split | Images |
|---|---:|
| Train | 1,766 |
| Validation | 379 |
| Test | 379 |
| **Total** | **2,524** |

Three duplicate image pairs were removed during dataset cleaning.

## 📈 Results

### Current Reproducible Run

| Metric | Result |
|---|---:|
| Best Validation Accuracy | **90.50%** |
| Test Accuracy | **88.65%** |
| Macro F1 | **0.8742** |

The repository also documents a **previous notebook run with 90.77% test accuracy**. That checkpoint was not retained after the original Colab session reset, so **88.65% is the current reproducible result**.

## 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **Torchvision**
- **Pillow**
- **Gradio**
- **Render**

## 📂 Project Structure

```text
Smart_WasteVision/
│
├── app.py
├── requirements.txt
├── .python-version
├── best_model_resnet18_weighted_finetuned.pth
├── Smart_WasteVision_Portfolio_clean.ipynb
└── README.md
```

### File Description

| File | Purpose |
|---|---|
| `app.py` | Gradio inference application |
| `requirements.txt` | Python dependencies |
| `.python-version` | Python version used for deployment |
| `best_model_resnet18_weighted_finetuned.pth` | Trained ResNet18 checkpoint |
| `Smart_WasteVision_Portfolio_clean.ipynb` | Training, evaluation, and experiment notebook |

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/tanim568/Smart_WasteVision.git
cd Smart_WasteVision
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the application

```bash
python app.py
```

The application will start locally using the Gradio interface.

## 🌐 Deployment

The application is deployed as a **Render Web Service**.

The application reads the `PORT` environment variable provided by Render and binds to `0.0.0.0` for production deployment.

**Live Demo:** https://smart-wastevision.onrender.com

## 📓 Notebook

The cleaned portfolio notebook contains the project workflow, including:

- Dataset preparation
- Data splitting
- Image preprocessing
- Model setup
- Fine-tuning
- Weighted loss
- Training and validation
- Evaluation metrics
- Classification report
- Confusion matrix
- Checkpoint reload verification

## 🎯 Project Highlights

This project demonstrates practical experience with:

- Computer Vision
- Image Classification
- Transfer Learning
- ResNet architectures
- Fine-tuning pretrained models
- Handling class imbalance
- Model evaluation
- PyTorch inference
- Gradio application development
- Cloud deployment

## 👤 Author

**Tanim Chy**

GitHub: https://github.com/tanim568

---

⭐ If you find this project useful, feel free to explore the repository and try the live demo.
