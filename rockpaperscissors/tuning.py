from itertools import product
from pathlib import Path
import csv, time, numpy as np, tensorflow as tf

from . import architectures, data_utils, training, config, evaluation

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
    model_names=("model_a", "model_b", "model_c"),   # <- niente model_d
    search_space=None,
    epochs=20,
    steps_train=None,   # es. 100 per velocizzare il tuning; None = full
    steps_val=None,     # es. 30
    checkpoint_dir="models",
    report_csv="reports/tuning_results.csv",
):
    """Esegue tuning su più modelli e iperparametri."""
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

            ckpt = f"{checkpoint_dir}/{name}_lr{lr}_b{batch}_aug{int(aug)}.keras"
            cbs = training.make_callbacks(checkpoint_path=ckpt)

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
                "n_params": n_params, "checkpoint": ckpt,
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
    """Training finale pulito sul best del tuning."""
    assert best is not None and "model_name" in best, "Best dict manca 'model_name'"

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

    final_model_path = f"models/{name}_final.keras"
    model.save(final_model_path)
    print(f"Saved final model to: {final_model_path}")

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

# ----------------- helpers per config / main -----------------
def _normalize_models(sel):
    alias = {"a":"model_a","b":"model_b","c":"model_c"}  # aggiungi "d" se lo implementi
    if isinstance(sel, str):
        tokens = [t.strip().lower() for t in sel.split(",") if t.strip()]
    elif isinstance(sel, (list, tuple, set)):
        tokens = [str(t).strip().lower() for t in sel if str(t).strip()]
    else:
        raise ValueError("TUNING_MODELS must be str or list/tuple/set")
    mapped = []
    for t in tokens:
        if t not in alias:
            raise ValueError(f"Invalid model alias: '{t}' (use a,b,c)")
        m = alias[t]
        if m not in mapped:
            mapped.append(m)
    if not mapped:
        raise ValueError("No valid model in TUNING_MODELS")
    return tuple(mapped)

def run_from_config():
    """Legge opzioni da config.py e lancia tuning + final training."""
    model_names = _normalize_models(getattr(config, "TUNING_MODELS", "c"))
    epochs_tune = getattr(
        config, "TUNING_EPOCHS",
        10 if getattr(config, "TUNING_FAST", False) else 20
    )
    steps_tr  = getattr(config, "TUNING_STEPS_TRAIN", None)  # << aggiunti
    steps_val = getattr(config, "TUNING_STEPS_VAL", None)    # << aggiunti

    # restringi lo spazio se solo C (veloce)
    search_space = {"lr":[3e-4,5e-4], "batch":[32,64], "augment":[True]} if model_names == ("model_c",) else None

    if bool(getattr(config, "NO_TUNING", False)):
        print("[tuning] NO_TUNING=True → salto la ricerca.")
        best = {"model_name": model_names[0],
                "lr": 3e-4 if model_names[0]=="model_c" else 1e-3,
                "batch": 32, "augment": True}
    else:
        best = run_tuning(model_names=model_names, search_space=search_space,
                          epochs=epochs_tune, steps_train=steps_tr, steps_val=steps_val)

    final_epochs = getattr(config, "FINAL_EPOCHS", 50)
    return run_final_training(best, epochs=final_epochs, checkpoint_path="models/final_best.keras")

# ----------------- CLI opzionale (tienine UNO solo) -----------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run tuning from CLI")
    p.add_argument("--models", type=str, default="c", help="a,b,c oppure 'b,c'")
    p.add_argument("--epochs", type=int, default=None, help="Epoche tuning")
    p.add_argument("--steps-train", type=int, default=None)
    p.add_argument("--steps-val", type=int, default=None)
    p.add_argument("--final-epochs", type=int, default=50)
    p.add_argument("--no-tuning", action="store_true")
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()

    # popola config e riusa run_from_config
    config.TUNING_MODELS = args.models
    config.TUNING_FAST = args.fast
    config.TUNING_EPOCHS = args.epochs if args.epochs is not None else (
        10 if getattr(config, "TUNING_FAST", False) else 20
    )
    config.TUNING_STEPS_TRAIN = args.steps_train
    config.TUNING_STEPS_VAL = args.steps_val
    config.FINAL_EPOCHS = args.final_epochs
    config.NO_TUNING = args.no_tuning

    run_from_config()