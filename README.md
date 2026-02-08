[README (4).md](https://github.com/user-attachments/files/25050081/README.4.md)
# CNN Image Classification – Statistical Learning (SoSe 2025)

**Authors:** Phillip Olshausen  
**Course:** Statistical Learning (Sommersemester 2025)  
**Type:** University project / GitHub portfolio submission

---

## 1. Project Motivation

This project implements a complete **image classification pipeline using Convolutional Neural Networks (CNNs)**.  
The primary goal is not only to achieve good predictive performance, but to **demonstrate a rigorous statistical learning workflow**:

- systematic data preprocessing  
- transparent model design  
- careful hyperparameter control  
- stratified cross-validation for robust performance estimation  
- comparison of different training strategies  
- critical evaluation of generalization and overfitting  

The notebook is intentionally written in a **clear, step-by-step, educational style**, making it suitable both for academic assessment and as a **portfolio project** showcasing applied machine learning skills.

---

## 2. Dataset Assumptions & Structure

The project assumes an **image dataset organized by class folders**, e.g.:

```
dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── class_k/
└── test/
    ├── class_1/
    ├── class_2/
    └── class_k/
```

Each class folder contains grayscale or RGB images belonging to the same category.

Key assumptions:
- Classes are mutually exclusive  
- Images may vary in size and are resized during preprocessing  
- Class imbalance is possible and explicitly checked  

---

## 3. Notebook Structure (Detailed Walkthrough)

### Section 1: CNN Model Pipeline

---

### Block 1 – Environment Setup & Library Imports

**Purpose:**  
Prepare a clean and reproducible environment.

**Key points:**
- TensorFlow logging is restricted to warnings/errors to avoid clutter
- All required libraries are imported in one place:
  - TensorFlow / Keras for modeling
  - NumPy for numerical operations
  - scikit-learn for data splitting and cross-validation
  - matplotlib for visualization

This block ensures that the rest of the notebook runs deterministically and transparently.

---

### Block 2 – Data Paths & Class Distribution

**Purpose:**  
Understand the dataset before modeling.

**What happens:**
- Training and test directory paths are defined
- Class labels are automatically inferred from directory names
- The number of images per class is counted and printed

**Why this matters:**
- Reveals potential **class imbalance**
- Prevents silent data leakage
- Ensures the label encoding is consistent and reproducible

---

### Block 3 – Hyperparameter Setup

**Purpose:**  
Centralize all experimental settings.

**Defined parameters include:**
- Image dimensions (height × width)
- Batch size
- Number of epochs
- Number of folds for cross-validation

**Why this matters:**
- Makes experiments easy to reproduce
- Allows controlled comparison of model variants
- Avoids hard-coded “magic numbers” scattered across the notebook

---

### Block 4 – Sample Image Visualization

**Purpose:**  
Perform a **visual sanity check** on the data.

**What happens:**
- One representative image per class is loaded
- Images are resized and converted to grayscale
- Samples are displayed side by side with class labels

**Why this matters:**
- Confirms correct data loading
- Reveals obvious labeling or preprocessing errors early
- Builds intuition about intra- and inter-class variability

---

### Block 5 – Data Loading & Normalization

**Purpose:**  
Transform raw image files into model-ready tensors.

**Steps:**
- Load images from disk
- Resize to target dimensions
- Convert to grayscale
- Normalize pixel values to the range \[0, 1\]
- One-hot encode class labels

**Output:**
- NumPy arrays for training and test data
- Consistent tensor shapes suitable for CNN input

---

### Block 6 – Original vs. Normalized Image Comparison

**Purpose:**  
Illustrate the effect of normalization.

**What is shown:**
- Raw grayscale image
- Corresponding normalized version

**Why this matters:**
- Demonstrates numerically stable input scaling
- Helps explain faster and more stable CNN training
- Useful for educational and presentation purposes

---

### Block 7 – CNN Definition & Stratified k-Fold Cross-Validation

**Purpose:**  
Estimate generalization performance robustly.

**Key components:**
- CNN architecture definition (convolution → pooling → dense → softmax)
- Stratified k-fold split on training data
- Separate model instantiation for each fold
- Recording best validation accuracy per fold

**Why stratification matters:**
- Preserves class proportions in each fold
- Reduces variance in performance estimates
- Particularly important for imbalanced datasets

**Outcome:**
- Mean validation accuracy
- Standard deviation across folds

This block provides the **statistically most reliable performance estimate** in the notebook.

---

### Block 8 – Full Training Without Hold-Out Validation

**Purpose:**  
Establish a baseline training strategy.

**What happens:**
- Model is trained on the full training dataset
- No validation split is used during training
- Final evaluation is performed on the test set

**Why include this:**
- Shows the risk of overfitting when no validation monitoring is used
- Serves as a comparison point for the next block

---

### Block 9 – Final Model Training With Hold-Out Validation

**Purpose:**  
Train the final model under best-practice conditions.

**Key aspects:**
- Fresh model initialization
- Stratified train/validation split
- Validation metrics monitored each epoch
- Learning curves plotted

**Why this matters:**
- Prevents overfitting
- Enables early diagnostics of under/over-parameterization
- Produces the final model suitable for reporting

---

## 4. Model Architecture (Conceptual)

The CNN follows a standard image-classification design:

- **Convolutional layers:** learn local spatial features  
- **Pooling layers:** reduce spatial resolution and variance  
- **Dense layers:** combine high-level features  
- **Softmax output:** produce class probabilities  

The architecture balances expressive power with overfitting control, making it appropriate for small to medium-sized datasets.

---

## 5. Evaluation Strategy

The project evaluates performance using:
- Cross-validated validation accuracy (mean ± standard deviation)
- Final test accuracy
- Training and validation learning curves

This combination allows:
- Robust performance estimation
- Detection of overfitting
- Transparent comparison of training strategies

---

## 6. Reproducibility

To reproduce results:
1. Clone the repository
2. Install dependencies
3. Adjust dataset paths in Block 2
4. Run the notebook top-to-bottom

All randomness is controlled where applicable, and hyperparameters are explicitly documented.

---

## 7. Intended Audience

This repository is suitable for:
- University coursework evaluation
- Machine learning portfolio review
- Demonstrating applied CNN workflows
- Readers seeking an interpretable, well-documented ML project

---

## 8. License & Disclaimer

This project is intended for **educational and portfolio use**.  
Dataset licensing depends on the original data source and is not included in this repository.

---

*Statistical Learning – Sommersemester 2025*
