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
├── rockpaperscissors/ # Core package
│ ├── architectures.py # CNN architectures (A, B, C, D)
│ ├── config.py # Global config (image size, batch size, seed, class names)
│ ├── data_utils.py # Data loading, cleaning, preprocessing, augmentation
│ ├── evaluation.py # Evaluation metrics, confusion matrices, misclassified samples
│ ├── training.py # Training loop and callbacks
│ └── init.py
│
├── notebooks/ # (Optional) Jupyter notebooks for EDA
├── main.py # Main script to train, evaluate, and generate reports
│
├── data/ # Dataset (rock/, paper/, scissors/)
├── reports/ # Plots, classification reports, confusion matrices
└── models/ # Saved trained models (.keras)
```

---
## 🧹 Dataset Preparation, Cleaning & Splitting

The dataset should follow this folder structure:
```
data/
├── rock/
├── paper/
└── scissors/
```

- Each folder contains images of the corresponding gesture.  
- An **external test set** can optionally be added in `data/rps-cv-images/`.

### Cleaning
- Images are automatically resized to `(128, 128)` pixels.  
- Pixel values are normalized in `[0, 1]`.  
- Label encoding is handled by TensorFlow’s `image_dataset_from_directory`.

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

The project trains and compares **four CNN architectures**:

- **`model_a`**:  
  Baseline CNN with a few convolutional + dense layers.  

- **`model_b`**:  
  Lightweight model using **SeparableConv2D** for efficiency.  

- **`model_c`**:  
  Residual CNN with projection shortcuts, **label smoothing**, and **dropout** for regularization.  

- **`model_d`**:  
  Deeper residual CNN, includes **LayerNorm**, dropout, and gradient stabilization.  

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

---

## 🔍 Hyperparameter Tuning

At the end of `main.py`, a **grid search** is implemented to optimize:
- **Learning rate**: `[1e-3, 5e-4, 3e-4]`  
- **Batch size**: `[16, 32]`  
- **Data augmentation**: `[True, False]`  

The script tests all combinations, logs validation accuracy, and prints the **best configuration**.
The best configuration is saved and reported in `reports/summary.csv`.

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
   - Train the four architectures 
   - Save trained models in ```/models/``` 
   - Generate reports and plots in ```/reports/```

3. Check outputs:
   - reports/summary.csv: validation results for each model 
   - Confusion matrices, learning curves, misclassified samples

--- 
✨ Notes

- Models are reproducible with ```config.SEED```. 
- Works on CPU, faster on GPU. 
- Architectures C and D integrate residual connections, dropout, and label smoothing to improve generalization.

