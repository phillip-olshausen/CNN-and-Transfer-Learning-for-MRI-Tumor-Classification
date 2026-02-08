[README_portfolio.md](https://github.com/user-attachments/files/25171015/README_portfolio.md)
# CNN Image Classification – Statistical Learning (SoSe 2025)

**Authors:** Phillip Olshausen  
**Course:** Data Science (Sommersemester 2025)  
**Type:** University project 

This repository contains a **fully documented Jupyter notebook** that implements and compares:
1) a **custom CNN trained from scratch**, and  
2) a **transfer learning model (EfficientNetB0)**

Both approaches are evaluated with **robust metrics, learning curves, and error analysis** (confusion matrices + misclassification inspection).  
> ✅ A detailed, block-by-block explanation of the full workflow is included directly inside the notebook.

---

## What this project does (in one sentence)

Given a folder-structured image dataset, we build a reproducible pipeline to **train, validate, and test** a CNN classifier, then **benchmark it against transfer learning** to analyze generalization and failure cases.

---

## Key Features

- **Two modeling paradigms**
  - **Custom CNN** (from scratch)
  - **Transfer Learning with EfficientNetB0** (pretrained backbone + custom head)
- **Clean preprocessing pipeline**
  - resizing, grayscale handling (Section 1)
  - **contrast enhancement / histogram equalization** visualization (Section 2)
  - normalization to stable numeric ranges
- **Robust evaluation**
  - stratified splits and **Stratified K-Fold Cross-Validation** (Section 1)
  - hold-out validation monitoring for the final run
  - test-set evaluation
- **Interpretability & error analysis**
  - learning curves (loss/accuracy)
  - **confusion matrices**
  - most frequent misclassifications
  - visual inspection of misclassified examples
  - confidence inspection (probability gaps)
- **Extra (portfolio-friendly)**
  - interactive single-image prediction demo for the transfer learning model

---

## Repository Contents

- **Notebook:** `LiviaKastrati_PhillipOlshausen_StatLearningSoSe2025.ipynb`  
  The complete project implementation including explanations, figures, training logs, and evaluation.

> Tip for a professional repo: place the notebook under `notebooks/` and keep results plots under `figures/`.

---

## Dataset Assumptions

The notebook expects an image dataset arranged by **class folders**, typically:

```text
data/
├── train/
│   ├── class_A/
│   ├── class_B/
│   └── ...
└── test/
    ├── class_A/
    ├── class_B/
    └── ...
```

- **Class names** are auto-detected from folder names.
- The notebook prints **class counts** to detect imbalance early.
- Images are resized to a fixed shape defined in the hyperparameters.

> Note: The dataset itself is typically not included in public repos due to size/licensing.  
> Add `data/` to `.gitignore` and document download instructions in `data/README.md`.

---

## Notebook Walkthrough (Complete)

## Section 1 — Custom CNN Model (from scratch)

This section implements a classic statistical learning workflow with a CNN baseline and rigorous evaluation.

### Block 1 — Environment Setup & Imports
- Configure TensorFlow logging
- Import ML + plotting + evaluation dependencies

### Block 2 — Data Paths & Class Distribution
- Set dataset directories
- Auto-detect classes
- Print per-class sample counts (train/test)

### Block 3 — Hyperparameter Setup
- Image size, batch size, epochs
- CV folds (Stratified K-Fold)
- Seeds / reproducibility settings where relevant

### Block 4 — Sample Image Visualization
- Display one example per class (sanity check)

### Block 5 — Data Loading & Normalization
- Load all images into arrays
- Convert to grayscale (if configured)
- Normalize pixel values to **[0, 1]**
- One-hot encode labels

### Block 6 — Original vs Normalized Comparison
- Visual explanation of why normalization improves training stability

### Block 7 — CNN Definition + Stratified K-Fold Cross-Validation
- Define a CNN architecture (conv → pooling → dense → softmax)
- Evaluate with **stratified folds** for robust performance estimates
- Report mean ± std validation accuracy across folds

### Block 8 — Full Training (No Hold-Out)
- Train on full training set without validation monitoring
- Baseline to compare against validation-monitored training

### Block 9 — Final Training with Hold-Out Validation
- Stratified train/val split
- Train a fresh CNN
- Track validation curves for overfitting diagnostics

### Blocks 10–17 — Evaluation + Diagnostics
- Test set evaluation
- Metrics reporting and overfitting checks
- Learning curves (accuracy/loss)
- Confusion matrix
- Misclassification patterns and examples
- Confidence inspection for wrong predictions

---

## Section 2 — Transfer Learning Benchmark (EfficientNetB0)

This section repeats the pipeline with a modern pretrained backbone and directly compares it to the custom CNN.

### Block 1 — Imports & Settings
- Centralized configuration for the TL experiment

### Block 2 — Auto-detect Class Names
- Ensures label mapping is identical to folder structure

### Block 3 — Count & List Example Files
- Quickly verify data integrity per class

### Block 4 — Visualization: Original → Equalized
- Shows preprocessing/contrast enhancement effects (useful for model robustness)

### Block 5 — Load & Preprocess Functions
- Unified preprocessing for both models to ensure a fair comparison

### Block 6 — Load Train & Test Data
- Load full train/test arrays with consistent processing

### Block 7 — Validation Split
- Separate hold-out validation set for monitoring (transfer learning training stability)

### Block 8A — Build Custom CNN (baseline)
- Rebuild a scratch CNN as the comparison baseline within this section

### Block 8B — Build Transfer Learning Model (EfficientNetB0)
- Load EfficientNetB0 pretrained weights
- Attach a classification head
- Train with fine-tuning strategy as configured

### Block 9 — Train Both Models
- Train baseline CNN and TL model under comparable conditions
- Store training history for curve plotting

### Block 10 — Evaluate on Test Set
- Report test performance for both models

### Block 11 — Plot Training Curves
- Compare convergence and generalization dynamics

### Block 12 — Confusion Matrices
- Side-by-side confusion matrices to interpret where each model fails

### Block 13 — Misclassified Examples
- Visualize wrong predictions for qualitative insights

### Block 14 (Extra) — Interactive Single Image Prediction (TL model)
- User-facing inference demo: input an image → get predicted class + confidence

---

## Results (How to present this on GitHub)

In your GitHub portfolio, it helps to include a short, visual summary in the README:

- Cross-validation accuracy (Section 1)
- Final test accuracy (CNN vs EfficientNetB0)
- One confusion matrix image
- A few representative misclassifications

**Suggested placeholders (fill with your numbers):**
- **Custom CNN:** CV acc ≈ `__` (mean ± std), Test acc = `__`
- **EfficientNetB0 TL:** Test acc = `__` (and/or val acc = `__`)

---

## How to Run (Reproducibility)

1. Create/activate an environment
2. Install dependencies
3. Set dataset paths
4. Run the notebook top-to-bottom

### Dependencies
- Python 3.x
- TensorFlow / Keras
- NumPy, scikit-learn, matplotlib
- (Optional) OpenCV / image utilities depending on your preprocessing blocks

### Minimal setup
```bash
pip install -r requirements.txt
```

---



## License / Notes
This project is intended for educational and portfolio use. Dataset inclusion depends on the original data source and licensing.

---

**Notebook:**  
✅ Contains full documented explanations, code blocks, and results.
ADME_portfolio.md…]()
