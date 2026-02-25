# Rock-Paper-Scissors CNN Classifier 

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow 2.12](https://img.shields.io/badge/TensorFlow-2.12-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras)](https://keras.io/)

This project implements **Convolutional Neural Networks (CNNs)** to classify images of hand gestures representing the **Rock–Paper–Scissors** game.  
The codebase is designed to be **modular, reproducible, and extensible**, including data preprocessing, model architectures, training loops, evaluation metrics, and hyperparameter tuning.

---
## 📂 Project Structure
```
rockpaperscissor_ml_project/
│
├── rockpaperscissors/                  # Core package
│   ├── architectures.py                # CNN architectures (A, B, C)
│   ├── config.py                       # Global config (image size, batch size, seed, class names, tuning flags)
│   ├── data_utils.py                   # Data loading, cleaning, preprocessing, augmentation
│   ├── evaluation.py                   # Evaluation metrics, confusion matrices, misclassified samples
│   ├── training.py                     # Training loop and callbacks
│   ├── tuning.py                       # Hyperparameter tuning (CLI + run_from_config used by main.py)
│   ├── visualize_models.py             # png of best models architectures
│   └── __init__.py                     
│
├── analysis/                           # Explanatory Data Analysis
├── main.py                             # Main script to train, evaluate, and generate reports
│
├── data/                               # Dataset (rock/, paper/, scissors/)
├── reports/                            # Plots, classification reports, confusion matrices, tuning CSV
├── models/                             # Saved trained models (.keras)
├── project_report/                     # project report in LaTeX
├── evaluate_myhands.py                 # Evaluate on custom “my hands” images
├── external_eval.py                    # Evaluate on external test "rps-cv-images" images
├── inspect_model.py                    # (Utility) Model inspection / summaries / parameter counts
└── requirements.txt                    

```

---
## 🧹 Dataset Preparation, Cleaning & Splitting

The dataset is available on Kaggle:  
👉 [Rock-Paper-Scissors Dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)

Steps to prepare the dataset:
1. Download the dataset from Kaggle.  
2. Extract the archive into the `data/` folder so that the structure looks like this:  
```
data/
├── rock/
├── paper/
└── scissors/
```

- Each folder contains images of the corresponding gesture.  
- An **external test set** was added in `data/rps-cv-images/`.

### Cleaning
- Images are automatically resized to **(IMG_SIZE, IMG_SIZE)** as defined in `config.py` (`96×96`)  
- Pixel values are normalized in `[0, 1]`.  
- Training/validation labels are manually encoded, while the external test set uses TensorFlow’s `image_dataset_from_directory`.

### Splitting
- The dataset is split into **training (80%)** and **validation (20%)**.  
- Stratified sampling ensures class balance.  
- External test images are never seen during training.

### Data Augmentation
When enabled, the training set undergoes lightweight augmentation:
- Random horizontal flips  
- Random rotations (±10%)  
- Random zoom  

---

## 🏗️ Model Architectures

The project trains and compares **three CNN architectures**:

- **`model_a`**:  
  Baseline CNN with a few convolutional + dense layers.  

- **`model_b`**:  
  Lightweight model using **SeparableConv2D** for efficiency.  

- **`model_c`**:  
  Residual CNN with projection shortcuts, **label smoothing**, and **dropout** for regularization.  

All models use:
- **Adam optimizer** (default LR `3e-4`)  
- **Categorical cross-entropy** loss (with label smoothing)  
- Accuracy as the main metric  

---

## ⚙️ Training Pipeline

1. **Data loading** → handled by `data_utils.py`.  
2. **Model selection** → defined in `architectures.py`.  
3. **Training loop** → controlled by `training.py` with callbacks:
   - EarlyStopping (patience on validation loss)  
   - ReduceLROnPlateau (LR scheduling)  
   - ModelCheckpoint (best model saving)  

4. **Evaluation** → executed in `evaluation.py`, producing:
   - Classification report (`precision`, `recall`, `f1-score`)  
   - Confusion matrix plots  
   - Training curves (loss & accuracy)  
   - Grids of **most confident misclassified images**  

   **Generalization Test with custom images**
   To further evaluate the **generalization capability** of the models, we tested them on a **custom dataset** of hand gesture photos.
   The images were organized into subfolders:
   ```
   my_hands/
   ├── rock/
   ├── paper/
   └── scissors/
   ```
   Run the evaluation with:

   ```bash
   python evaluate_myhands.py --dir data/my_hands_data --model models/model_a_best.keras  --outdir reports\custom_eval_myhands\myhands_model_a
   python evaluate_myhands.py --dir data/my_hands_data --model models/model_b_best.keras  --outdir reports\custom_eval_myhands\myhands_model_b
   python evaluate_myhands.py --dir data/my_hands_data --model models/model_c_best.keras  --outdir reports\custom_eval_myhands\myhands_model_c
   ```
   This generates:
   - custom_classification_report.txt → precision, recall, f1-score 
   - custom_confusion_matrix.png → confusion matrix on the custom dataset 
   - custom_misclassified.png → a grid of misclassified examples (if any)

---

## 🔍 Hyperparameter Tuning

Tuning is handled via `rockpaperscissors/tuning.py`.

### Option A — Run from CLI
**`CLI` example**
```bash
# Model C only, 12 epochs
python -m rockpaperscissors.tuning --models c --epochs 12

# Models B and C, fast preset, limited steps per trial
python -m rockpaperscissors.tuning --models b,c --fast --steps-train 120 --steps-val 40
```
### Option B — Run from `main.py` via config

Enable tuning from your project’s `config.py`, then run `main.py`. The main will short-circuit into the tuner and stop after the final training of the best config.

**`config.py` example**
```python
TUNING = True                     # enable tuning path
TUNING_MODELS = "c"               # e.g., "b,c" or ["a","b","c"]
TUNING_EPOCHS = 12                # default 20 (or 10 if TUNING_FAST=True)
TUNING_FAST = False               # optional shortcut: if True and TUNING_EPOCHS not set, uses 10
TUNING_STEPS_TRAIN = None         # e.g., 120 to speed up each trial
TUNING_STEPS_VAL = None           # e.g., 40
FINAL_EPOCHS = 50                 # epochs for final training on the best model
```
---

## ▶️ How to Run

1. **Install requirements**:

    ```bash
    pip install -r requirements.txt
    ```
   Main requirements:
    - tensorflow >= 2.12 
    - numpy 
    - matplotlib 
    - scikit-learn 
    - pillow

2. **Train and evaluate models**:

    ```bash
    python main.py
    ```
   - Train the three architectures 
   - Save trained models in ```/models/``` 
   - Generate reports and plots in ```/reports/```

3. **Run external evaluation only on in-distribution dataset**:

    ```bash
    python external_eval.py
    ```

   - Load a previously trained final model
   - Evaluate it on an external dataset (`rps-cv-images`)
   - Generate external classification report and confusion matrix
   - Does **not** retrain or modify the models


--- 
## 📁 Reports & Outputs

The project intentionally separates training, tuning, evaluation, and inspection into different scripts.
As a result, different subfolders inside ```reports/``` are generated by different entry points.

This section clarifies exactly which script produces which outputs.

### 1️⃣ python main.py — Standard Training & Validation
Running:
```bash
python main.py
```
(with  ```TUNING=False``` in  ```config.py ```) produces:
```
reports/
├── model_a/
├── model_b/
├── model_c/
│   ├── fig_training_accuracy.png
│   ├── fig_training_loss.png
│   ├── val_classification_report.txt
│   ├── val_confusion_matrix.png
│   └── val_misclassified.png
│
├── summary.csv
```

Purpose
* Train each architecture on the training set
* Evaluate on the validation split
* Compare models in a single CSV summary

### 2️⃣ python main.py with Tuning Enabled

If in config.py:
```python
TUNING = True
```
then ```main.py``` delegates execution to the tuner and exits after final training.
This produces:
```
reports/
├── model_a_final/   (if selected)
├── model_b_final/
├── model_c_final/
│   ├── fig_training_accuracy.png
│   ├── fig_training_loss.png
│   ├── val_classification_report.txt
│   ├── val_confusion_matrix.png
│   └── val_misclassified.png
│
├── tuning_results.csv
```

### 3️⃣ python external_eval.py — External / In-Distribution Evaluation
> Note: This script assumes that a final trained model (e.g., model_c_final) already exists.

Running:
```bash
python external_eval.py 
```
produces:

```
reports/
├── model_c_final/
│   ├── test_classification_report.txt
│   └── test_confusion_matrix.png
```
Purpose
* Load the final (best) trained model
* Evaluate it on an external dataset (rps-cv-images)
* Perform post-hoc external evaluation only

### 4️⃣ python evaluate_myhands.py — Custom “My Hands” Evaluation
Running:
```bash
python evaluate_myhands.py --dir data/my_hands_data --model models/model_a_best.keras 
```
produces:
```
reports/
└── custom_eval_myhands/
    └── my_hands_model_a_best/
        ├── classification_report_model_a_best.txt
        ├── confusion_matrix_model_a_best.png
        ├── misclassified_model_a_best.png
        └── summary_model_a_best.txt
```

Purpose
* Evaluate generalization on custom, real-world images
* Use labeled folder structure as ground truth

### 5️⃣ python inspect_model.py — Model Inspection 
Running: 
```bash
python inspect_model.py models/model_c_best.keras 
```
produces:
```
reports/
└── model_c_final/   (or model_c/)
    ├── inspect_confusion_matrix_best.png
    └── classification_report_best.txt
```

Purpose
* Inspect a saved model without retraining
* Print model summary and parameter count
* Recompute validation performance

---

## 🗂️ Model Files (`.keras`)

The trained models are saved in the `.keras` format (introduced in Keras 3).  
Each file contains:

- the **architecture** (layers, activations, etc.)  
- the **trained weights**  
- the **optimizer state** (useful to resume training)  

These files are **binary** and cannot be previewed on GitHub, but can be reloaded in Python.

### Generate Architecture Diagrams

This project includes a utility to export PNG diagrams of saved Keras models.

Requirements:
- Graphviz installed and available in PATH.

```bash
export PATH="$PATH:/c/Program Files/Graphviz/bin"
python -m rockpaperscissors.visualize_models
```

-----
> *Future Work suggestions*
> - Extend dataset with more hand gestures;
> - Test the use of pretrained CNN models to improve generalization;
> - Deploy as a web app for interactive play.
