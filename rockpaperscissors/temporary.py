import tensorflow as tf
from pathlib import Path

from rockpaperscissors import config, data_utils, evaluation


def regenerate(model_name: str, batch_size: int = 32):
    model_path = f"models/{model_name}_best.keras"
    out_dir = Path(f"reports/{model_name}_final")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Regenerating artifacts for {model_name} ===")

    # IMPORTANT: validation must be deterministic
    _, val_ds, file_paths_val = data_utils.load_train_val_stratified(
        validation_split=0.2,
        augment=False,   # NEVER use augmentation in evaluation
        batch_size=batch_size,
        cache_train=False,
        cache_val=False,
    )

    model = tf.keras.models.load_model(model_path)

    # Single pass evaluation (CM + probs + y_true)
    res = evaluation.evaluate_on(val_ds, model, config.CLASSES)

    # Confusion matrix
    evaluation.plot_confusion(
        res["cm"],
        config.CLASSES,
        outpath=str(out_dir / "val_confusion_matrix.png"),
        title=f"Confusion Matrix (val) – {model_name}",
    )

    # Misclassified grid (guaranteed consistent with CM)
    evaluation.save_misclassified_from_probs(
        file_paths=file_paths_val,
        probs=res["probs"],
        y_true=res["y_true"],
        class_names=config.CLASSES,
        top_n=12,
        pick="confident",
        outpath=str(out_dir / "val_misclassified.png"),
    )

    # Optional: classification report
    evaluation.save_report(
        res["report_txt"],
        str(out_dir / "val_classification_report.txt"),
    )

    print(f"[OK] Saved in {out_dir}")
    print(f"Val accuracy: {res['acc']:.4f}")


def main():
    regenerate("model_a")
    regenerate("model_b")
    regenerate("model_c")


if __name__ == "__main__":
    main()