from rockpaperscissors import config, data_utils, architectures, training, evaluation
import numpy as np, tensorflow as tf, random, os
from pathlib import Path
from itertools import product
import csv

# Reproducibility
tf.random.set_seed(config.SEED); np.random.seed(config.SEED); random.seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)
Path("models").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True)


def train_and_report(model_name, model, train_ds, val_ds, file_paths_val):
    model.summary()

    history, runtime = training.train(
        model, train_ds, val_ds, epochs=30,
        callbacks=training.make_callbacks(checkpoint_path=f"models/{model_name}_best.keras")
    )
    print(f"Total training time: {runtime:.1f}s | Avg/epoch: {runtime/len(history.history['loss']):.2f}s")
    best_epoch = 1 + int(np.argmin(history.history["val_loss"]))
    print(f"Early stopped at epoch {best_epoch} (best val_loss={min(history.history['val_loss']):.4f})")
    model.save(f"models/{model_name}.keras")

    # Validation metrics
    res_val = evaluation.evaluate_on(val_ds, model, config.CLASSES)
    print(f"Validation accuracy: {res_val['acc']:.4f}")

    # Reports & plots
    evaluation.save_report(res_val["report_txt"], f"reports/{model_name}_val_classification_report.txt")
    evaluation.plot_confusion(res_val["cm"], config.CLASSES, f"reports/{model_name}_val_confusion_matrix.png",
                              title=f"Confusion Matrix (val) – {model_name}")
    evaluation.plot_history(history, outdir=f"reports/{model_name}")

    # Misclassified: most confident errors
    try:
        evaluation.show_misclassified(
            val_ds, model, file_paths_val, config.CLASSES,
            top_n=12, outpath=f"reports/{model_name}_val_misclassified.png", pick="confident"
        )
    except Exception as e:
        print("Impossible to generate misclassified grid:", e)

    n_params = model.count_params()
    return res_val['acc'], runtime, n_params, float(min(history.history["val_loss"]))


def main():
    # Data
    train_ds, val_ds, file_paths_val = data_utils.load_train_val(validation_split=0.2, augment=True)

    # === Four architectures ===
    results = []
    for name, builder in [
        ("model_a", architectures.model_a),
        ("model_b", architectures.model_b),
        ("model_c", architectures.model_c),
        ("model_d", architectures.model_d),
    ]:
        tf.keras.backend.clear_session()
        acc, runtime, n_params, best_vloss = train_and_report(name, builder(), train_ds, val_ds, file_paths_val)
        results.append((name, acc, runtime, n_params, best_vloss))

    # Table for report summary
    with open("reports/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "val_accuracy", "train_runtime_s", "params", "best_val_loss"])
        w.writerows(results)

    # === External test on the best model ===
    try:
        best_name = max(results, key=lambda t: t[1])[0]
        best_model = tf.keras.models.load_model(f"models/{best_name}.keras")
        test_ds = data_utils.load_external_test()
        res_test = evaluation.evaluate_on(test_ds, best_model, config.CLASSES)
        evaluation.save_report(res_test["report_txt"], f"reports/{best_name}_test_classification_report.txt")
        evaluation.plot_confusion(res_test["cm"], config.CLASSES, f"reports/{best_name}_test_confusion_matrix.png",
                                  title=f"Confusion Matrix (external test) – {best_name}")
    except Exception as e:
        print("No external test set found:", e)


if __name__ == "__main__":
    main()

    # Hyperparameter tuning
    search_space = {
        "lr": [1e-3, 5e-4, 3e-4],
        "batch": [16, 32],
        "augment": [True, False],
    }
    best = None
    for lr, batch, aug in product(search_space["lr"], search_space["batch"], search_space["augment"]):
        print(f"\n=== Trying lr={lr}, batch={batch}, augment={aug} ===")
        # load data with the parameters
        train_ds, val_ds, _ = data_utils.load_train_val(validation_split=0.2, augment=aug, batch_size=batch)
        model = architectures.model_a()
        model.optimizer.learning_rate = lr

        history, runtime = training.train(model, train_ds, val_ds, epochs=20)
        val_acc = max(history.history["val_accuracy"])
        cand = {"lr": lr, "batch": batch, "augment": aug, "val_acc": float(val_acc)}
        if best is None or val_acc > best["val_acc"]:
            best = cand
        print("Current best:", best)
    print("=== BEST CONFIG ===", best)
