import argparse
import os
from pathlib import Path
import tensorflow as tf
import numpy as np

from rockpaperscissors import config, evaluation


def _ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def _artifact_tag(model_path: str) -> str:
    """
    Crea un tag parlante dai nomi dei file modello:
      models/model_a_best.keras -> 'model_a_best'
      models/model_a.keras      -> 'model_a'
      models/final_best.keras   -> 'final_best'
      qualunque_altro           -> stem del file
    """
    stem = Path(model_path).stem
    return stem  # così avrai esattamente 'model_a', 'model_a_best', 'final_best', ecc.


def load_labeled_dir(root_dir, batch_size=None):
    """Carica le immagini (ordine classi fissato) e normalizza x/255."""
    bs = batch_size or config.BATCH_SIZE
    ds_raw = tf.keras.utils.image_dataset_from_directory(
        root_dir,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=bs,
        label_mode="categorical",
        shuffle=False,
        class_names=list(config.CLASSES),
    )
    print("Using class order:", ds_raw.class_names)

    ds = ds_raw.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).cache().prefetch(tf.data.AUTOTUNE)

    file_paths = ds_raw.file_paths
    return ds, file_paths


def main(model_path: str, data_dir: str, outdir: str):
    tag = _artifact_tag(model_path)

    print(f"🔍 Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"📁 Evaluating on labeled folder: {data_dir}")
    ds, file_paths = load_labeled_dir(data_dir)

    # Metrics and evaluation
    res = evaluation.evaluate_on(ds, model, config.CLASSES)
    print(f"\nAccuracy: {res['acc']:.4f}\n")
    print(res["report_txt"])

    # Analisi dettagliata
    probs_all, ytrue_all = [], []
    for xb, yb in ds:
        p = model.predict(xb, verbose=0)
        probs_all.append(p)
        ytrue_all.append(yb.numpy())
    probs = np.concatenate(probs_all, axis=0)
    y_true = np.argmax(np.concatenate(ytrue_all, axis=0), axis=1)
    y_pred = np.argmax(probs, axis=1)

    print("Unique y_true:", np.unique(y_true), "Unique y_pred:", np.unique(y_pred))
    print("Pred distrib (counts):", np.bincount(y_pred, minlength=len(config.CLASSES)))
    print("True  distrib (counts):", np.bincount(y_true, minlength=len(config.CLASSES)))
    print("Mean prob per class:", probs.mean(axis=0))
    print("Max prob (mean±std):", probs.max(axis=1).mean(), probs.max(axis=1).std())

    # Top-5 'scissors'
    top_sc_idx = np.argsort(probs[:, config.CLASSES.index("scissors")])[::-1][:5]
    print("\nTop-5 'scissors' prob:")
    for i in top_sc_idx:
        print(f"{file_paths[i]}  ->  probs={probs[i]}")

    # Salvataggi con nomi univoci basati sul tag
    _ensure_dir(outdir)
    report_path = os.path.join(outdir, f"classification_report_{tag}.txt")
    cm_path     = os.path.join(outdir, f"confusion_matrix_{tag}.png")
    mis_path    = os.path.join(outdir, f"misclassified_{tag}.png")

    evaluation.save_report(res["report_txt"], report_path)
    evaluation.plot_confusion(
        res["cm"], config.CLASSES, cm_path,
        title=f"Confusion Matrix – {tag.replace('_', ' ')}"
    )
    print(f"✅ Saved report to {report_path}")
    print(f"✅ Saved confusion matrix to {cm_path}")

    # Misclassified
    try:
        try:
            evaluation.show_misclassified(
                ds, model, file_paths, config.CLASSES,
                top_n=12, outpath=mis_path, pick="confident"
            )
        except TypeError:
            evaluation.show_misclassified(
                ds, model, file_paths, config.CLASSES,
                top_n=12, outpath=mis_path
            )
        print(f"🖼️  Saved misclassified grid to {mis_path}")
    except Exception as e:
        print(f"(info) Unable to save misclassified grid: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved .keras model on your labeled hand-gesture photos."
    )
    parser.add_argument("--model", required=True,
                        help="Path to .keras model (e.g., models/model_a_best.keras)")
    parser.add_argument("--dir", default="my_hands",
                        help="Root folder with subfolders rock/paper/scissors (default: my_hands)")
    parser.add_argument("--outdir", default="reports/custom_eval_myhands",
                        help="Where to save reports/plots (default: reports/custom_eval_myhands)")
    args = parser.parse_args()

    main(args.model, args.dir, args.outdir)
