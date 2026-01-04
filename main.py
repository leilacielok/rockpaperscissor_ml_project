import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from rockpaperscissors import config, tuning, data_utils, architectures, training, evaluation
import numpy as np, tensorflow as tf, random
from pathlib import Path
from itertools import product
import csv

# Reproducibility
tf.random.set_seed(config.SEED); np.random.seed(config.SEED); random.seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)
Path("models").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True)


def train_and_report(model_name, model, train_ds, val_ds, file_paths_val):
    model.summary()

    ckpt_path = f"models/{model_name}_best.keras"
    history, runtime = training.train(
        model, train_ds, val_ds, epochs=30,
        callbacks=training.make_callbacks(checkpoint_path=ckpt_path)
    )
    print(f"Total training time: {runtime:.1f}s | Avg/epoch: {runtime/len(history.history['loss']):.2f}s")
    best_epoch = 1 + int(np.argmin(history.history.get('val_loss', history.history['loss'])))
    best_vloss = float(min(history.history.get('val_loss', [np.inf])))
    print(f"Early stopped at epoch {best_epoch} (best val_loss={best_vloss:.4f})")
    model.save(f"models/{model_name}.keras")

    # folder for reports
    model_dir = Path("reports") / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Validation metrics
    res_val = evaluation.evaluate_on(val_ds, model, config.CLASSES)
    print(f"Validation accuracy: {res_val['acc']:.4f}")

    # Reports & plots
    evaluation.save_report(
        res_val["report_txt"],
        str(model_dir / "val_classification_report.txt")
    )
    evaluation.plot_confusion(
        res_val["cm"], config.CLASSES,
        outpath=str(model_dir / "val_confusion_matrix.png"),
        title=f"Confusion Matrix (val) – {model_name}"
    )
    evaluation.plot_history(history, outdir=str(model_dir))

    # Misclassified: most confident errors
    try:
        evaluation.show_misclassified(
            val_ds, model, file_paths_val, config.CLASSES,
            top_n=12, outpath=str(model_dir/"val_misclassified.png"), pick="confident"
        )
    except Exception as e:
        print("Impossible to generate misclassified grid:", e)

    n_params = model.count_params()
    return res_val['acc'], runtime, n_params, best_vloss


def main():
    # tuning
    if getattr(config, "TUNING", False):
        tuning.run_from_config()
        raise SystemExit(0)
    # Data
    train_ds, val_ds, file_paths_val = data_utils.load_train_val_stratified(
        validation_split=0.2, augment=True
    )
    evaluation.print_class_histogram(val_ds)

    # Priors
    priors = data_utils.compute_class_priors(train_ds, len(config.CLASSES))
    log_priors = np.log(priors + 1e-8)
    print("class priors:", priors)

    models_to_try = [
        ("model_a", architectures.model_a),
        ("model_b", architectures.model_b),
        ("model_c", lambda: architectures.model_c(log_priors)),
    ]

    # Four architectures
    results = []
    for name, builder in models_to_try:
        tf.keras.backend.clear_session()
        acc, runtime, n_params, best_vloss = train_and_report(
            name, builder(), train_ds, val_ds, file_paths_val
        )
        results.append((name, acc, runtime, n_params, best_vloss))

    # Table for report summary
    with open("reports/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "val_accuracy", "train_runtime_s", "params", "best_val_loss"])
        w.writerows(results)

    # External test on the best model
    try:
        best_name = max(results, key=lambda t: t[1])[0]
        from keras.models import load_model
        best_model = load_model(f"models/{best_name}.keras")
        test_ds = data_utils.load_external_test()
        res_test = evaluation.evaluate_on(test_ds, best_model, config.CLASSES)

        best_dir = Path("reports") / best_name
        best_dir.mkdir(parents=True, exist_ok=True)

        evaluation.save_report(
            res_test["report_txt"],
            str(best_dir / "test_classification_report.txt")
        )
        evaluation.plot_confusion(
            res_test["cm"], config.CLASSES,
            outpath=str(best_dir / "test_confusion_matrix.png"),
            title=f"Confusion Matrix (external test) – {best_name}"
        )
    except Exception as e:
        print("No external test set found:", e)

if __name__ == "__main__":
    main()