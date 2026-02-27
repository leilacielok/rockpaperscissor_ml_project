import csv
from pathlib import Path
import numpy as np
import tensorflow as tf

from rockpaperscissors import config, data_utils, evaluation


def evaluate_model(model_name: str, batch_size: int = 32):
    model_path = f"models/{model_name}_best.keras"

    # Validation set (NO augmentation)
    _, val_ds, _ = data_utils.load_train_val_stratified(
        validation_split=0.2,
        augment=False,
        batch_size=batch_size,
        cache_train=False,
        cache_val=False,
    )

    model = tf.keras.models.load_model(model_path)

    res = evaluation.evaluate_on(val_ds, model, config.CLASSES)

    # Confidence statistics
    probs = res["probs"]
    conf = probs.max(axis=1)

    return {
        "model": model_name,
        "val_accuracy": res["acc"],
        "precision_macro": res["report_dict"]["macro avg"]["precision"],
        "recall_macro": res["report_dict"]["macro avg"]["recall"],
        "f1_macro": res["report_dict"]["macro avg"]["f1-score"],
        "params": model.count_params(),
        "val_conf_mean": float(np.mean(conf)),
        "val_conf_std": float(np.std(conf)),
    }


def main():
    Path("reports").mkdir(exist_ok=True)

    models = ["model_a", "model_b", "model_c"]
    rows = [evaluate_model(m) for m in models]

    out_path = "reports/summary_best.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Saved {out_path}")


if __name__ == "__main__":
    main()