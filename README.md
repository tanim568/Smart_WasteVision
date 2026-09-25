# Smart WasteVision

This project classifies waste images into six TrashNet categories: cardboard, glass, metal, paper, plastic, and trash. The final model follows the weighted fine-tuned ResNet18 methodology described in the notebook and is deployed with a Gradio app.

## Previous result
Previous result: 90.77% test accuracy from the original notebook run.

## Current reproducible run
- Dataset size: 2524 total images
- Train / Validation / Test: 1766 / 379 / 379
- Model: ResNet18 with Dropout(0.4) and Linear(512, 6)
- Training epochs: 8
- Best validation accuracy: 90.50%
- Final test accuracy: 88.65%
- Macro F1: 0.8742
- Checkpoint size: 44,801,675 bytes (~42.7 MB)
- Deployment status: ready for Render Free Web Service on 0.0.0.0 using $PORT

## Files
- best_model_resnet18_weighted_finetuned.pth — trained checkpoint
- app.py — Gradio inference application
- Smart_WasteVision_Portfolio_clean.ipynb — cleaned portfolio notebook

## Run locally
```bash
pip install -r requirements.txt
python app.py
```

The app binds to 0.0.0.0 and reads the Render $PORT environment variable when deployed.
