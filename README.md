# Smart WasteVision

A deep learning image classification project for classifying waste images into six categories using **PyTorch, CNNs, and ResNet18 transfer learning**.

## Overview

Smart WasteVision explores waste-image classification using the TrashNet dataset. The project progresses from a custom CNN baseline to ResNet18 transfer learning, fine-tuning, and class-weighted fine-tuning.

The final evaluation reported in the portfolio notebook uses a **class-weighted, fine-tuned ResNet18**.

## Classes

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## Workflow

1. Dataset inspection
2. Duplicate-image detection
3. Class distribution analysis
4. Train/validation/test split
5. Image preprocessing and augmentation
6. Custom CNN experiments
7. ResNet18 frozen-backbone transfer learning
8. ResNet18 fine-tuning
9. Class-weighted fine-tuning
10. Validation-based model selection
11. Final test evaluation
12. Confusion matrix and error analysis
13. Single-image prediction
14. Gradio inference demo

## Model Results

| Experiment | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| ResNet18 — frozen backbone | 72.03% | 0.6584 |
| ResNet18 — fine-tuned | 90.24% | 0.8816 |
| ResNet18 — fine-tuned + class weights | 89.45% | 0.8750 |

The final reported test evaluation was performed with the class-weighted fine-tuned ResNet18.

## Final Test Results

**Test Accuracy: 90.77%**

**Test Macro F1: 0.91**

**Test Samples: 379**

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Cardboard | 0.98 | 0.92 | 0.95 |
| Glass | 0.94 | 0.84 | 0.89 |
| Metal | 0.79 | 0.93 | 0.86 |
| Paper | 0.92 | 0.96 | 0.94 |
| Plastic | 0.89 | 0.89 | 0.89 |
| Trash | 1.00 | 0.90 | 0.95 |

### Confusion Matrix

Rows represent the true class and columns represent the predicted class.

| True \ Predicted | Cardboard | Glass | Metal | Paper | Plastic | Trash |
|---|---:|---:|---:|---:|---:|---:|
| Cardboard | 55 | 0 | 1 | 4 | 0 | 0 |
| Glass | 0 | 64 | 4 | 1 | 7 | 0 |
| Metal | 0 | 2 | 57 | 1 | 1 | 0 |
| Paper | 1 | 0 | 3 | 85 | 0 | 0 |
| Plastic | 0 | 2 | 5 | 1 | 64 | 0 |
| Trash | 0 | 0 | 2 | 0 | 0 | 19 |

The largest visible confusion patterns are **glass → plastic (7)**, **plastic → metal (5)**, **glass → metal (4)**, and **cardboard → paper (4)**.

## Dataset

The project uses the **TrashNet** dataset and its resized images.

- Total images inspected: **2,527**
- Exact duplicate images found: **3**
- Classes: **6**

The dataset has relatively controlled image backgrounds and capture conditions. Performance on more varied real-world photographs has not been established by this project.

## Error Analysis

The test errors show that visually similar materials can be difficult to separate. In particular, glass, metal, and plastic account for several of the observed confusions. Some objects can also have visual characteristics that overlap with another material category.

## Gradio Demo

The repository includes a simple Gradio application for single-image prediction.

The app:

- loads the fine-tuned ResNet18 architecture
- applies the same ImageNet normalization used for inference
- returns the top-3 predicted classes
- accepts an uploaded image through a Gradio interface

The trained model checkpoint is intentionally **not included in GitHub** because of its file size.

### Run the demo locally

1. Install the dependencies.
2. Place the model checkpoint in the project root:

`best_model_resnet18_weighted_finetuned.pth`

3. Run:

```bash
python app.py
```

The model checkpoint must be available locally before starting the app.

## Project Structure

```text
Smart_WasteVision/
├── Smart_WasteVision_Portfolio.ipynb
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```bash
git clone https://github.com/tanim568/Smart_WasteVision.git
cd Smart_WasteVision
pip install -r requirements.txt
```

Then open `Smart_WasteVision_Portfolio.ipynb` in Google Colab or Jupyter.

## Limitations

- The dataset is relatively small.
- The classes are imbalanced, especially `trash`.
- The images have relatively similar capture conditions.
- Visually similar materials can be difficult to distinguish.
- Each experiment was trained once using a fixed split.
- The trained model weights are not included in the repository.

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Gradio
- Google Colab / Jupyter

## Author

**Tanim Chy**

GitHub: https://github.com/tanim568
