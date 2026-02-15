import csv
import time
from itertools import product
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import architectures, config, data_utils, evaluation, training


# ----------------- builders -----------------
def _build_model(name, log_priors=None):
    if name == "model_a":
        return architectures.model_a()
    if name == "model_b":
        return architectures.model_b()
    if name == "model_c":
        return architectures.model_c(log_priors=log_priors)
    raise ValueError(f"Unknown model name: {name}")

# ----------------- tuning core -----------------
def run_tuning(
    model_names=("model_a", "model_b", "model_c"),
    search_space=None,
    epochs=20,
    steps_train=None,   # es. 100 to speed up tuning; None = full
    steps_val=None,     # es. 30
    checkpoint_dir="models",
    report_csv="reports/tuning_results.csv",
):
    if search_space is None:
        search_space = {"lr": [1e-3, 5e-4, 3e-4], "batch": [16, 32], "augment": [True, False]}

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(report_csv).parent).mkdir(parents=True, exist_ok=True)

    results, best = [], None

    for name in model_names:
        for lr, batch, aug in product(search_space["lr"], search_space["batch"], search_space["augment"]):
            print(f"\n=== Trying {name} | lr={lr}, batch={batch}, augment={aug} ===")

            train_ds, val_ds, _ = data_utils.load_train_val_stratified(
                validation_split=0.2, augment=aug, batch_size=batch
            )

            priors = data_utils.compute_class_priors(train_ds, len(config.CLASSES))
            log_priors = np.log(priors + 1e-8)

            tf.keras.backend.clear_session()
            model = _build_model(name, log_priors=log_priors)
            cbs = training.make_callbacks(checkpoint_path=None)

            try:
                model.optimizer.learning_rate = lr
            except Exception:
                pass

            train_input = train_ds.take(steps_train) if steps_train else train_ds
            val_input   = val_ds.take(steps_val)     if steps_val   else val_ds

            t0 = time.time()
            history, runtime = training.train(model, train_input, val_input, epochs=epochs, callbacks=cbs)
            wall = time.time() - t0

            val_acc = float(max(history.history["val_accuracy"]))
            n_params = int(model.count_params())

            row = {
                "model_name": name, "lr": lr, "batch": batch, "augment": aug,
                "val_acc": val_acc, "runtime_fit": float(runtime), "runtime_wall": float(wall),
                "n_params": n_params, "checkpoint": None,
            }
            results.append(row)
            if best is None or val_acc > best["val_acc"]:
                best = row
            print("Current best:", best)

    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("=== BEST CONFIG ===", best)
    print(f"Report saved to: {report_csv}")
    return best

# ----------------- final train -----------------
def run_final_training(
    best: dict,
    epochs: int = 50,
    checkpoint_path: str = "models/final_best.keras",
    reports_dir: str = "reports",
):
    """Final training on best model after tuning."""
    assert best is not None and "model_name" in best, "Best dict miss 'model_name'"

    name   = best["model_name"]; lr = float(best["lr"])
    batch  = int(best["batch"]); augment = bool(best["augment"])

    Path("models").mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n=== FINAL TRAINING === {name} | lr={lr} | batch={batch} | augment={augment}\n")

    train_ds, val_ds, file_paths_val = data_utils.load_train_val_stratified(
        validation_split=0.2, augment=augment, batch_size=batch
    )
    priors = data_utils.compute_class_priors(train_ds, len(config.CLASSES))
    log_priors = np.log(priors + 1e-8)

    tf.keras.backend.clear_session()
    model = _build_model(name, log_priors=log_priors)

    cbs = training.make_callbacks(checkpoint_path=checkpoint_path)
    try:
        model.optimizer.learning_rate = lr
    except Exception:
        pass

    history, runtime = training.train(model, train_ds, val_ds, epochs=epochs, callbacks=cbs, learning_rate=lr)
    print(f"Final training time: {runtime:.1f}s | Best val_acc: {max(history.history['val_accuracy']):.4f}")

    model_dir = Path(reports_dir) / f"{name}_final"
    model_dir.mkdir(parents=True, exist_ok=True)

    res_val = evaluation.evaluate_on(val_ds, model, config.CLASSES)
    print(f"Final VAL accuracy: {res_val['acc']:.4f}")

    evaluation.save_report(res_val["report_txt"], str(model_dir / "val_classification_report.txt"))
    evaluation.plot_confusion(res_val["cm"], config.CLASSES,
                              outpath=str(model_dir / "val_confusion_matrix.png"),
                              title=f"Confusion Matrix (val) – {name} (final)")
    evaluation.plot_history(history, outdir=str(model_dir))

    try:
        evaluation.show_misclassified(
            val_ds, model, file_paths_val, config.CLASSES,
            top_n=12, outpath=str(model_dir/"val_misclassified.png"), pick="confident"
        )
    except Exception as e:
        print("Impossible to generate misclassified grid:", e)

    return model, history, runtime, res_val

# ----------------- helpers for config / main -----------------
def _map_model_token(token):
    """Map short alias (a,b,c) or full name to canonical model name."""
    mapping = {
        "a": "model_a",
        "b": "model_b",
        "c": "model_c",
        "model_a": "model_a",
        "model_b": "model_b",
        "model_c": "model_c",
    }

    token = str(token).strip().lower()
    if token not in mapping:
        raise ValueError(f"Invalid model name: {token}")
    return mapping[token]

def main():
    model_names = [_map_model_token(m) for m in config.TUNING_MODELS]
    best = run_tuning(
            model_names=model_names,
            epochs=config.TUNING_EPOCHS,
            steps_train=config.TUNING_STEPS_TRAIN,
            steps_val=config.TUNING_STEPS_VAL,
        )
    ckpt = f"models/{best['model_name']}_best.keras"

    run_final_training(best, epochs=config.FINAL_EPOCHS, checkpoint_path=ckpt)

if __name__ == "__main__":
    main()
