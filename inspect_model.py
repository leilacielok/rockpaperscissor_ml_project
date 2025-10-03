import tensorflow as tf
import sys
from rockpaperscissors import config, data_utils, evaluation

def main(model_path):
    print(f"🔍 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 1) Print summary
    print("\n=== Model Summary ===")
    model.summary()

    # 2) Print number of parameters
    print(f"\nTotal trainable params: {model.count_params():,}")

    # 3) Optional: evaluate on validation set
    print("\n=== Validation Performance ===")
    _, val_ds, _ = data_utils.load_train_val_stratified(validation_split=0.2, augment=False)
    res = evaluation.evaluate_on(val_ds, model, config.CLASSES)
    print(f"Validation Accuracy: {res['acc']:.4f}")
    print("\nClassification Report:\n", res["report_txt"])

    # 4) Save confusion matrix
    evaluation.plot_confusion(
        res["cm"], config.CLASSES,
        outpath="reports/inspect_confusion_matrix.png",
        title=f"Confusion Matrix – {model_path}"
    )
    print("Confusion matrix saved to reports/inspect_confusion_matrix.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py <path_to_model.keras>")
    else:
        main(sys.argv[1])
