import argparse
import os
from pathlib import Path
import tensorflow as tf
import numpy as np

from rockpaperscissors import config, evaluation

def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_labeled_dir(root_dir, batch_size=None):
    """Load the images from root_dir with subfolders = classes."""
    bs = batch_size or config.BATCH_SIZE
    ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
        root_dir,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=bs,
        label_mode="categorical",
        shuffle=False,
        class_names=config.CLASSES,
    )
    file_paths = ds_raw.file_paths
    norm = tf.keras.layers.Rescaling(1.0/255.0)
    ds = ds_raw.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache().prefetch(tf.data.AUTOTUNE)
    return ds, file_paths

def main(model_path: str, data_dir: str, outdir: str):
    print(f"🔍 Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"📁 Evaluating on labeled folder: {data_dir}")
    ds, file_paths = load_labeled_dir(data_dir)

    # Metrics and evaluation
    res = evaluation.evaluate_on(ds, model, config.CLASSES)
    print(f"\nAccuracy: {res['acc']:.4f}\n")
    print(res["report_txt"])

    # raccogliamo probs e y_true per analisi
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

    # Mostra le 5 immagini più 'scissors' secondo il modello
    top_sc_idx = np.argsort(probs[:, config.CLASSES.index("scissors")])[::-1][:5]
    print("\nTop-5 'scissors' prob:")
    for i in top_sc_idx:
        print(f"{file_paths[i]}  ->  probs={probs[i]}")

   # Savings
    _ensure_dir(outdir)
    report_path = os.path.join(outdir, "custom_classification_report.txt")
    cm_path = os.path.join(outdir, "custom_confusion_matrix.png")
    evaluation.save_report(res["report_txt"], report_path)
    evaluation.plot_confusion(res["cm"], config.CLASSES, cm_path, title="Confusion Matrix – Custom Set")
    print(f"✅ Saved report to {report_path}")
    print(f"✅ Saved confusion matrix to {cm_path}")

    # Misclassified
    try:
        try:
            evaluation.show_misclassified(
                ds, model, file_paths, config.CLASSES,
                top_n=12, outpath=os.path.join(outdir, "custom_misclassified.png"), pick="confident"
            )
        except TypeError:
            evaluation.show_misclassified(
                ds, model, file_paths, config.CLASSES,
                top_n=12, outpath=os.path.join(outdir, "custom_misclassified.png")
            )
        print(f"🖼️  Saved misclassified grid to {os.path.join(outdir, 'custom_misclassified.png')}")
    except Exception as e:
        print(f"(info) Unable to save misclassified grid: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved .keras model on your labeled hand-gesture photos."
    )
    parser.add_argument("--model", default="models/model_d.keras",
                        help="Path to .keras model (default: models/model_d.keras)")
    parser.add_argument("--dir", default="my_hands",
                        help="Root folder with subfolders rock/paper/scissors (default: my_hands)")
    parser.add_argument("--outdir", default="reports/custom_eval_myhands",
                        help="Where to save reports/plots (default: reports/custom_eval_myhands)")
    args = parser.parse_args()

    main(args.model, args.dir, args.outdir)
