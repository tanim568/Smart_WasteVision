# Smart WasteVision

A deep learning image classification project for classifying waste images into six categories using CNNs and ResNet18 transfer learning.

## Overview

Smart WasteVision compares multiple image-classification approaches on the TrashNet dataset, progressing from a custom CNN to ResNet18 transfer learning and fine-tuning.

The final reported model is a **class-weighted, fine-tuned ResNet18** selected using validation performance.

## Classes

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## Project Workflow

1. Dataset inspection and cleaning
2. Duplicate-image detection
3. Class distribution analysis
4. Stratified train/validation/test split
5. Image preprocessing and augmentation
6. Custom CNN experiments
7. ResNet18 transfer learning
8. ResNet18 fine-tuning
9. Class-weighted fine-tuning
10. Validation-based model selection
11. Final test evaluation
12. Confusion matrix and error analysis
13. Single-image prediction

## Model Comparison

| Experiment | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|
| Custom CNN, standard loss | 63.85% | 0.5324 |
| Custom CNN, class-weighted loss | 57.26% | 0.5435 |
| ResNet18, frozen backbone | 72.03% | 0.6584 |
| ResNet18, fine-tuned | 90.24% | 0.8816 |
| ResNet18, fine-tuned + class-weighted | 89.18% | **0.8862** |

The class-weighted fine-tuned ResNet18 was selected based on validation Macro F1. Class weighting mainly improved recall for the smaller **trash** class.

## Final Test Results

**Class-Weighted Fine-Tuned ResNet18**

- **Test Accuracy:** 87.86%
- **Test Macro F1:** 0.8837
- **Test Samples:** 379
- **Misclassified samples:** 46

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Cardboard | 0.96 | 0.90 | 0.93 |
| Glass | 0.95 | 0.74 | 0.83 |
| Metal | 0.73 | 0.95 | 0.83 |
| Paper | 0.92 | 0.92 | 0.92 |
| Plastic | 0.84 | 0.89 | 0.86 |
| Trash | 0.95 | 0.90 | 0.93 |

## Error Analysis

There were **46 incorrect predictions** in the final test set.

The most frequent confusion patterns were:

- **Glass → Metal:** 11
- **Glass → Plastic:** 8
- **Plastic → Metal:** 5
- **Cardboard → Paper:** 4

These errors mainly occurred between visually similar materials or objects.

## Dataset

The project uses the **TrashNet** dataset and its dataset-resized images.

- Raw images inspected: **2,527**
- Exact duplicate images found: **3**
- Images remaining after duplicate removal: **2,524**
- Classes: **6**

The dataset contains relatively controlled image backgrounds and capture conditions, so performance on cluttered real-world photographs has not been established by this project.

## Limitations

- The dataset is relatively small.
- The classes are imbalanced, especially `trash`.
- Images have relatively similar capture conditions.
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
- Google Colab / Jupyter

## Project Structure

- `Smart_WasteVision_Portfolio.ipynb` — cleaned portfolio notebook
- `README.md` — project documentation
- `requirements.txt` — Python dependencies

## How to Run

1. Clone the repository.
2. Install the dependencies from `requirements.txt`.
3. Open `Smart_WasteVision_Portfolio.ipynb` in Google Colab or Jupyter.
4. Prepare/download the TrashNet dataset as described in the notebook.
5. Run the notebook from the beginning.

Training the models again may produce slightly different results because of training randomness.

## Key Takeaway

The project demonstrates the progression from a custom CNN baseline to transfer learning and fine-tuning with ResNet18. The final reported class-weighted fine-tuned ResNet18 achieved **87.86% test accuracy** and **0.8837 test Macro F1** on 379 held-out test images.

## Author

**Tanim Chy**

GitHub: [tanim568](https://github.com/tanim568)