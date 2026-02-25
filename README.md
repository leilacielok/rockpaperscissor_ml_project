# Rock-Paper-Scissors CNN Classifier 

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.20](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
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
│   ├── tuning.py                       # Hyperparameter tuning (config-driven)
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

## 🚀 Installation
```bash
pip install -r requirements.txt
```

## ▶️ Quick Start (full pipeline)
```bash
python main.py
```
This will: 
- Train the three architectures 
- Evaluate on the validation split
- Select the best model and evaluate it on the external test set
- Save trained models in ```/models/``` 
- Generate reports and plots in ```/reports/```

---
## 🧹 Dataset Preparation & Preprocessing

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

- An **external test set** was added in `data/rps-cv-images/`.
- Optional **custom real-world dataset** (`my_hands_data`): manually collected set of hand gesture images organized as:
  ```
  data/
  └── my_hands_data/
      ├── rock/
      ├── paper/
      └── scissors/

### Cleaning
- Images are automatically resized to **(IMG_SIZE, IMG_SIZE)** as defined in `config.py` (`96×96`)  
- Pixel values are normalized in `[0, 1]`.  
- Training/validation labels are manually encoded, while the external test set uses TensorFlow’s `image_dataset_from_directory`.

### Splitting
- The dataset is split into **training (80%)** and **validation (20%)**.  
- Stratified sampling ensures class balance.  
- External test images are never seen during training.

### Data Augmentation
When enabled, the training set undergoes lightweight (random) augmentation:
- horizontal flips  
- rotations (±0.08 of a full turn ≈ ±29°)  
- zoom  (±10%)
- translations (up to 8% in both directions)
- contrast adjustment (±5%)

---

## 🏗️ Model Architectures

The project trains and compares **three CNN architectures**:

- **`model_a`**:  
  Lightweight CNN based on SeparableConv2D layers and Global Average Pooling for parameter efficiency. 

- **`model_b`**:  
  Conventional shallow CNN using standard Conv2D layers followed by flattening and a dense classifier.

- **`model_c`**:  
  Residual CNN with identity shortcut connections, depthwise separable blocks, and dropout for regularization. 

### 🔧 Training Configuration
All models use:
- Adam optimizer 
- Categorical cross-entropy with label smoothing
- Accuracy as main metric  
- EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## 🔍 External & Generalization Evaluation
1. Structured external test set
  `rps-cv-images` loaded via `image_dataset_from_directory`

2. Custom real-world dataset
  `my_hands_data` loaded via a custom `tf.data` pipeline with:
  - PIL decoding
  - EXIF correction
  - Resize with padding
  - Manual one-hot encoding

Run
```bash
python evaluate_myhands.py --dir data/my_hands_data --model models/model_x_best.keras
```
---
## 🔍 Hyperparameter Tuning

Tuning is handled via `rockpaperscissors/tuning.py`.
The search space (learning rate, batch size, data augmentation) is defined in `config.py`.

### Run tuning directly
```bash
python -m rockpaperscissors.tuning 
```

### Or enable via config

Set ```TUNING = True```, then:
```bash
python main.py
```

--- 
## 📁 Reports & Outputs

The project intentionally separates training, tuning, evaluation, and inspection into different scripts.
As a result, different subfolders inside ```reports/``` are generated by different entry points.

This section clarifies exactly which script produces which outputs.

### 1️⃣ Main script: Standard Training & Validation
Running:
```bash
python main.py
```
(with  ```TUNING=False``` in  ```config.py ```) produces:
```
reports/
├── model_x/
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

### 2️⃣ Main script with Tuning Enabled

(with  ```TUNING=True``` in  ```config.py ```), ```main.py``` delegates execution to the tuner and exits after final training.
This produces:
```
reports/
├── model_x_final/
│   ├── fig_training_accuracy.png
│   ├── fig_training_loss.png
│   ├── val_classification_report.txt
│   ├── val_confusion_matrix.png
│   └── val_misclassified.png
│
├── tuning_results.csv
```

### 3️⃣ External Evaluation on `rps-cv-images`
> Note: This script assumes that a final trained model (e.g., model_x_final) already exists.

```bash
python external_eval.py 
```
produces:

```
reports/
├── model_x_final/
│   ├── test_classification_report.txt
│   └── test_confusion_matrix.png
```
Purpose
* Load the final (best) trained model
* Evaluate it on an external dataset
* Perform post-hoc external evaluation only

### 4️⃣ External Evaluation on Custom Dataset
Running:
```bash
python evaluate_myhands.py --dir data/my_hands_data --model models/model_x_best.keras 
```
produces:
```
reports/
└── custom_eval_myhands/
    └── my_hands_model_x_best/
        ├── classification_report_model_x_best.txt → precision, recall, f1-score 
        ├── confusion_matrix_model_x_best.png
        ├── misclassified_model_x_best.png
        └── summary_model_x_best.txt
```

Purpose
* Evaluate generalization on custom, real-world images
* Use labeled folder structure as ground truth

### 5️⃣ Model Inspection 
```bash
python inspect_model.py models/model_x_best.keras 
```
produces:
```
reports/
└── model_x_final/   (or model_c/)
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
- the **optimizer state** 

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
