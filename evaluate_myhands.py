import argparse
import os
from pathlib import Path
import tensorflow as tf

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
        class_names=config.CLASSES,  # forza l'ordine coerente
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

    # Valutazione e metriche
    res = evaluation.evaluate_on(ds, model, config.CLASSES)
    print(f"\nAccuracy: {res['acc']:.4f}\n")
    print(res["report_txt"])

    # Salvataggi
    _ensure_dir(outdir)
    report_path = os.path.join(outdir, "custom_classification_report.txt")
    cm_path = os.path.join(outdir, "custom_confusion_matrix.png")
    evaluation.save_report(res["report_txt"], report_path)
    evaluation.plot_confusion(res["cm"], config.CLASSES, cm_path, title="Confusion Matrix – Custom Set")
    print(f"✅ Saved report to {report_path}")
    print(f"✅ Saved confusion matrix to {cm_path}")

    # Griglia dei misclassificati (se disponibile la funzione compatibile)
    try:
        # Alcune versioni della tua funzione hanno param 'pick'; se non c'è, ignora.
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
