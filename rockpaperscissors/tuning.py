# rockpaperscissors/tuning.py
from itertools import product
from pathlib import Path
import csv, time, numpy as np, tensorflow as tf

from . import architectures, data_utils, training, config

def _build_model(name, log_priors=None):
    if name == "model_a":
        return architectures.model_a()
    if name == "model_b":
        return architectures.model_b()
    if name == "model_c":
        return architectures.model_c(log_priors=log_priors)
    raise ValueError(f"Unknown model name: {name}")

def run_tuning(
    model_names=("model_a", "model_b", "model_c", "model_d"),
    search_space=None,
    epochs=20,
    steps_train=None,  # es. 100 per velocizzare il tuning; None = full
    steps_val=None,    # es. 30
    checkpoint_dir="models",
    report_csv="reports/tuning_results.csv",
):
    """
    Esegue tuning su più modelli e iperparametri.
    Ritorna un dict 'best' con: model_name, lr, batch, augment, val_acc, runtime, n_params, checkpoint.
    """
    if search_space is None:
        search_space = {
            "lr":    [1e-3, 5e-4, 3e-4],
            "batch": [16, 32],
            "augment": [True, False],
        }

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(report_csv).parent).mkdir(parents=True, exist_ok=True)

    results = []
    best = None

    for name in model_names:
        for lr, batch, aug in product(search_space["lr"], search_space["batch"], search_space["augment"]):
            print(f"\n=== Trying {name} | lr={lr}, batch={batch}, augment={aug} ===")

            # Dati con parametri correnti
            train_ds, val_ds, _ = data_utils.load_train_val_stratified(
                validation_split=0.2, augment=aug, batch_size=batch
            )

            # Priors per C/D (ok ricalcolarli per trial)
            n_classes = len(config.CLASSES)
            priors = data_utils.compute_class_priors(train_ds, n_classes)
            log_priors = np.log(priors + 1e-8)

            tf.keras.backend.clear_session()
            model = _build_model(name, log_priors=log_priors)

            # Checkpoint “parlante”
            ckpt = f"{checkpoint_dir}/{name}_lr{lr}_b{batch}_aug{int(aug)}.keras"
            cbs = training.make_callbacks(checkpoint_path=ckpt)

            # Aggiorna LR scelto per il trial
            try:
                model.optimizer.learning_rate = lr
            except Exception:
                # se non fosse compilato, training.train lo ricompila comunque
                pass

            # Se vuoi limitare il costo per trial
            train_input = train_ds.take(steps_train) if steps_train else train_ds
            val_input   = val_ds.take(steps_val)     if steps_val   else val_ds

            t0 = time.time()
            history, runtime = training.train(
                model, train_input, val_input,
                epochs=epochs, callbacks=cbs
            )
            wall = time.time() - t0

            val_acc = float(max(history.history["val_accuracy"]))
            n_params = int(model.count_params())

            row = {
                "model_name": name,
                "lr": lr,
                "batch": batch,
                "augment": aug,
                "val_acc": val_acc,
                "runtime_fit": float(runtime),
                "runtime_wall": float(wall),
                "n_params": n_params,
                "checkpoint": ckpt,
            }
            results.append(row)

            if best is None or val_acc > best["val_acc"]:
                best = row

            print("Current best:", best)

    # Salva report CSV
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("=== BEST CONFIG ===", best)
    print(f"Report saved to: {report_csv}")
    return best

def run_final_training(
    best: dict,
    epochs: int = 50,
    checkpoint_path: str = "models/final_best.keras",
    reports_dir: str = "reports",
):
    """
    Esegue il training finale 'pulito' usando il best dict del tuning.
    - Ricostruisce il modello vincente
    - Carica i dati con batch/augment migliori
    - Imposta il learning rate migliore
    - Salva modello, checkpoint e report/plot di validazione
    Ritorna (model, history, runtime, res_val)
    """
    assert best is not None and "model_name" in best, "Best dict manca 'model_name' (hai lanciato run_tuning?)"

    name   = best["model_name"]
    lr     = float(best["lr"])
    batch  = int(best["batch"])
    augment = bool(best["augment"])

    Path("models").mkdir(parents=True, exist_ok=True)
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n=== FINAL TRAINING on BEST ===\nModel: {name} | lr={lr} | batch={batch} | augment={augment}\n")

    # Dati con i parametri migliori
    train_ds, val_ds, file_paths_val = data_utils.load_train_val_stratified(
        validation_split=0.2, augment=augment, batch_size=batch
    )

    # Priors per C/D
    priors = data_utils.compute_class_priors(train_ds, len(config.CLASSES))
    log_priors = np.log(priors + 1e-8)

    tf.keras.backend.clear_session()
    model = _build_model(name, log_priors=log_priors)

    # Callbacks e LR finale
    cbs = training.make_callbacks(checkpoint_path=checkpoint_path)
    try:
        model.optimizer.learning_rate = lr
    except Exception:
        pass

    history, runtime = training.train(
        model, train_ds, val_ds,
        epochs=epochs, callbacks=cbs, learning_rate=lr
    )
    print(f"Final training time: {runtime:.1f}s | Best val_acc: {max(history.history['val_accuracy']):.4f}")

    # Salva modello completo
    final_model_path = f"models/{name}_final.keras"
    model.save(final_model_path)
    print(f"Saved final model to: {final_model_path}")

    # Report/plot di validazione
    model_dir = Path(reports_dir) / f"{name}_final"
    model_dir.mkdir(parents=True, exist_ok=True)

    res_val = evaluation.evaluate_on(val_ds, model, config.CLASSES)
    print(f"Final VAL accuracy: {res_val['acc']:.4f}")

    evaluation.save_report(
        res_val["report_txt"],
        str(model_dir / "val_classification_report.txt")
    )
    evaluation.plot_confusion(
        res_val["cm"], config.CLASSES,
        outpath=str(model_dir / "val_confusion_matrix.png"),
        title=f"Confusion Matrix (val) – {name} (final)"
    )
    evaluation.plot_history(history, outdir=str(model_dir))

    # Misclassified più “sicuri”
    try:
        evaluation.show_misclassified(
            val_ds, model, file_paths_val, config.CLASSES,
            top_n=12, outpath=str(model_dir/"val_misclassified.png"), pick="confident"
        )
    except Exception as e:
        print("Impossible to generate misclassified grid:", e)

    return model, history, runtime, res_val


if __name__ == "__main__":
    # Esempio di uso stand-alone:
    # 1) tuning (riduci steps_* per tuning più rapido)
    best = run_tuning(
        model_names=("model_a","model_b","model_c"),
        epochs=20,
        steps_train=None,
        steps_val=None,
    )
    # 2) training finale sul best
    run_final_training(best, epochs=50, checkpoint_path="models/final_best.keras")
