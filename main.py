import csv
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from rockpaperscissors import (
    architectures,
    config,
    data_utils,
    evaluation,
    training,
    tuning,
)

# Reproducibility
tf.random.set_seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)


def write_best_summary(out_csv="reports/summary_best.csv"):
    Path("reports").mkdir(exist_ok=True)

    # validation set deterministico, senza augmentation
    _, val_ds, _ = data_utils.load_train_val_stratified(
        validation_split=0.2,
        augment=False,
    )

    rows = []
    for name in ["model_a", "model_b", "model_c"]:
        ckpt = f"models/{name}_best.keras"
        model = tf.keras.models.load_model(ckpt)

        res = evaluation.evaluate_on(val_ds, model, config.CLASSES)
        rep = res["report_dict"]
        macro = rep["macro avg"]

        rows.append([
            name,
            float(rep["accuracy"]),
            float(macro["precision"]),
            float(macro["recall"]),
            float(macro["f1-score"]),
            int(model.count_params()),
            float(res["val_conf_mean"]),
            float(res["val_conf_std"]),
        ])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model",
            "val_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "params",
            "val_conf_mean",
            "val_conf_std",
        ])
        w.writerows(rows)

    print(f"Wrote {out_csv}")


def train_and_report(model_name, model, train_ds, val_ds, file_paths_val):
    model.summary()

    ckpt_path = f"models/{model_name}_best.keras"
    history, runtime = training.train(
        model,
        train_ds,
        val_ds,
        epochs=30,
        callbacks=training.make_callbacks(checkpoint_path=ckpt_path),
    )
    print(
        f"Total training time: {runtime:.1f}s | Avg/epoch: {runtime/len(history.history['loss']):.2f}s"
    )
    best_epoch = 1 + int(
        np.argmin(history.history.get("val_loss", history.history["loss"]))
    )
    best_vloss = float(min(history.history.get("val_loss", [np.inf])))
    print(f"Early stopped at epoch {best_epoch} (best val_loss={best_vloss:.4f})")
    model.save(f"models/{model_name}.keras")

    # folder for reports
    model_dir = Path("reports") / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Validation metrics
    res_val = evaluation.evaluate_on(val_ds, model, config.CLASSES)
    rep = res_val["report_dict"]
    macro_prec = float(rep["macro avg"]["precision"])
    macro_rec  = float(rep["macro avg"]["recall"])
    macro_f1   = float(rep["macro avg"]["f1-score"])
    acc_rep    = float(rep["accuracy"])  

    print(
        f"VAL macro | acc={acc_rep:.4f} P={macro_prec:.4f} R={macro_rec:.4f} F1={macro_f1:.4f}"
    )
    # Reports & plots
    evaluation.save_report(
        res_val["report_txt"], str(model_dir / "val_classification_report.txt")
    )
    evaluation.plot_confusion(
        res_val["cm"],
        config.CLASSES,
        outpath=str(model_dir / "val_confusion_matrix.png"),
        title=f"Confusion Matrix (val) – {model_name}",
    )
    evaluation.plot_history(history, outdir=str(model_dir))

    # Misclassified: most confident errors
    try:
        evaluation.show_misclassified(
            val_ds,
            model,
            file_paths_val,
            config.CLASSES,
            top_n=12,
            outpath=str(model_dir / "val_misclassified.png"),
            pick="confident",
        )
    except Exception as e:
        print("Impossible to generate misclassified grid:", e)

    n_params = model.count_params()
    return res_val["acc"], runtime, n_params, best_vloss


def main():

    if getattr(config, "MAKE_BEST_SUMMARY", False):
        write_best_summary()
        return

    # tuning
    if getattr(config, "TUNING", False):
        tuning.main()
        raise SystemExit(0)
    # Data
    train_ds, val_ds, file_paths_val = data_utils.load_train_val_stratified(
        validation_split=0.2, augment=True
    )
    counts = evaluation.print_class_histogram(val_ds)
    print("Validation class histogram:", counts)
    
    # Priors
    priors = data_utils.compute_class_priors(train_ds, len(config.CLASSES))
    log_priors = np.log(priors + 1e-8)
    print("class priors:", priors)

    models_to_try = [
        ("model_a", architectures.model_a),
        ("model_b", architectures.model_b),
        ("model_c", lambda: architectures.model_c(log_priors)),
    ]

    # three architectures
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
        w.writerow(
            ["model", "val_accuracy", "train_runtime_s", "params", "best_val_loss"]
        )
        w.writerows(results)

    # External test on the best model
    try:
        best_name = max(results, key=lambda t: t[1])[0]
        best_model = tf.keras.models.load_model(f"models/{best_name}_best.keras")
        test_ds = data_utils.load_external_test()
        res_test = evaluation.evaluate_on(test_ds, best_model, config.CLASSES)

        best_dir = Path("reports") / best_name
        best_dir.mkdir(parents=True, exist_ok=True)

        evaluation.save_report(
            res_test["report_txt"], str(best_dir / "test_classification_report.txt")
        )
        evaluation.plot_confusion(
            res_test["cm"],
            config.CLASSES,
            outpath=str(best_dir / "test_confusion_matrix.png"),
            title=f"Confusion Matrix (external test) – {best_name}",
        )
    except Exception as e:
        print("No external test set found:", e)


if __name__ == "__main__":
    main()
