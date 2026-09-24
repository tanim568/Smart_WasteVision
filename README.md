# Smart WasteVision

A deep learning image classification project for classifying waste images into six categories using a custom CNN and ResNet18 transfer learning.

## Overview

Smart WasteVision explores several CNN-based approaches for waste classification and compares their performance on the TrashNet dataset. The project progresses from a custom CNN baseline to transfer learning and fine-tuning with ResNet18.

The final selected experiment uses a **weighted, fine-tuned ResNet18**.

## Classes

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## Project Workflow

1. Dataset inspection and class distribution analysis
2. Image quality and duplicate checking
3. Train/validation/test preparation
4. Custom CNN experiments
5. ResNet18 transfer learning
6. ResNet18 fine-tuning
7. Class-weighted fine-tuning
8. Model comparison
9. Final test evaluation
10. Confusion matrix and error analysis
11. Single-image prediction

## Experiments

| Experiment | Validation Accuracy |
|---|---:|
| Custom CNN baseline | 66.23% |
| ResNet18 transfer learning | 72.03% |
| Fine-tuned ResNet18 | 90.24% |
| Weighted fine-tuned ResNet18 | 89.18% |

The final model was selected based on the validation results and class-wise performance, with class weighting mainly improving the minority `trash` class recall.

## Final Test Results

**Weighted Fine-Tuned ResNet18**

- **Test Accuracy:** 87.86%
- **Test Macro F1:** 0.8837
- **Test Samples:** 379

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Cardboard | 0.96 | 0.90 | 0.93 |
| Glass | 0.95 | 0.74 | 0.83 |
| Metal | 0.73 | 0.95 | 0.83 |
| Paper | 0.92 | 0.92 | 0.92 |
| Plastic | 0.84 | 0.89 | 0.86 |
| Trash | 0.95 | 0.90 | 0.93 |

## Confusion Matrix

The final confusion matrix shows that most predictions were correct. The most frequent confusion was between **glass and metal**, followed by **glass and plastic**.

## Error Analysis

The final test set contained **46 wrong predictions**.

Common error patterns included:

- Glass → Metal: 11
- Glass → Plastic: 8
- Plastic → Metal: 5
- Cardboard → Paper: 4

Visual inspection showed that some waste objects have similar appearance or ambiguous material cues, which can make classification difficult.

## Dataset

This project uses the **TrashNet** dataset. The dataset contains six waste categories and images captured under relatively controlled conditions.

## Limitations

- The dataset is relatively small.
- Images have relatively similar capture conditions.
- Some classes have fewer examples than others.
- Visually similar materials can be difficult to distinguish.
- The final model weights are not included in this repository.

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PIL
- Google Colab

## Project Structure

- `Smart_WasteVision_Portfolio.ipynb` — cleaned project notebook
- `README.md` — project documentation

## How to Run

1. Open the notebook in Google Colab or a local Jupyter environment.
2. Install the required Python packages.
3. Prepare/download the TrashNet dataset.
4. Run the notebook from the beginning.
5. Follow the experiment sections to reproduce the model comparison and evaluation.

Training the models again may produce slightly different results because of training randomness.

## Key Takeaway

The experiments show a clear improvement from a custom CNN baseline to transfer learning and fine-tuning with ResNet18. The final weighted fine-tuned model achieved **87.86% test accuracy** and a **0.8837 Macro F1** on the held-out test set.

## Author

**Tanim Chy**

GitHub: [tanim568](https://github.com/tanim568)
