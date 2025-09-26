from rockpaperscissors import config, data_utils, architectures, training, evaluation
import numpy as np, tensorflow as tf, random, os
from pathlib import Path

# Reproducibility
tf.random.set_seed(config.SEED); np.random.seed(config.SEED); random.seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)
Path("models").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True)

def main():
    # Data
    train_ds, val_ds = data_utils.load_train_val(validation_split=0.2, augment=True)

    # Choose model
    model = architectures.model_a()
    model.summary()

    # Train
    history, runtime = training.train(model, train_ds, val_ds, epochs=30)
    print(f"Total training time: {runtime:.1f}s | Avg/epoch: {runtime/len(history.history['loss']):.2f}s")
    best_epoch = 1 + int(np.argmin(history.history["val_loss"]))
    print(f"Early stopped at epoch {best_epoch} (best val_loss={min(history.history['val_loss']):.4f})")
    model.save("models/best_baseline.keras")

    # Validation metrics
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")

    # External test (optional)
    try:
        test_ds = data_utils.load_external_test()
        res = evaluation.evaluate_on(test_ds, model, config.CLASSES)
        print(f"External test accuracy: {res['acc']:.4f}")
        print(res["report_txt"])
        evaluation.plot_confusion(res["cm"], config.CLASSES, "reports/confusion_matrix.png")
    except Exception as e:
        print("No external test set found:", e)

    # Curves
    evaluation.plot_history(history, outdir="reports")

if __name__ == "__main__":
    main()

