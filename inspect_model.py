import tensorflow as tf
import sys
from pathlib import Path
from rockpaperscissors import config, data_utils, evaluation

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _detect_final_arch() -> str:
    """
    Tenta di inferire l'architettura del 'final_best.keras':
    1) se esiste una (e una sola) cartella reports/model_x_final → usa quella x
    2) altrimenti prova a leggere reports/tuning_results.csv per la riga con val_acc max
    3) fallback: 'final'
    """
    candidates = [("model_a", Path("reports/model_a_final")),
                  ("model_b", Path("reports/model_b_final")),
                  ("model_c", Path("reports/model_c_final"))]
    existing = [arch for arch, p in candidates if p.exists()]
    if len(existing) == 1:
        return existing[0]

    # prova con tuning_results.csv
    tr_csv = Path("reports/tuning_results.csv")
    if tr_csv.exists():
        try:
            import csv
            with tr_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                best = max(rows, key=lambda r: float(r.get("val_acc", 0.0)))
                arch = best.get("model_name", "")
                if arch in {"model_a", "model_b", "model_c"}:
                    return arch
        except Exception:
            pass

    return "final"  # fallback neutro

def _parse_model_tag(model_path: Path):
    """
    Restituisce (arch, tag, title_add, outdir):
      - models/model_a_best.keras → ("model_a", "best_tuning", "Best after tuning", reports/model_a)
      - models/final_best.keras   → ("model_c", "bestofbest", "Best-of-best (tuning)", reports/model_c_final)  # se arch=c
      - altro                     → (stem, "snapshot", "Snapshot", reports/stem)
    """
    stem = model_path.stem  # es. "model_a_best"
    if stem == "final_best":
        arch = _detect_final_arch()
        if arch in {"model_a", "model_b", "model_c"}:
            outdir = Path("reports") / f"{arch}_final"
        else:
            outdir = Path("reports") / "final"
        return arch, "bestofbest", "Best-of-best (tuning)", outdir

    if stem.startswith("model_") and stem.endswith("_best"):
        arch = "_".join(stem.split("_")[:2])  # "model_a_best" -> "model_a"
        outdir = Path("reports") / arch
        return arch, "best_tuning", "Best after tuning", outdir

    # fallback generico
    outdir = Path("reports") / stem
    return stem, "snapshot", "Snapshot", outdir

def main(model_path_str: str):
    model_path = Path(model_path_str)
    assert model_path.exists(), f"Model not found: {model_path}"

    print(f"🔍 Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    # 1) Summary
    print("\n=== Model Summary ===")
    model.summary()

    # 2) Numero parametri
    print(f"\nTotal trainable params: {model.count_params():,}")

    # 3) Valutazione su validation set (senza augmentation)
    print("\n=== Validation Performance ===")
    _, val_ds, _ = data_utils.load_train_val_stratified(
        validation_split=0.2, augment=False, batch_size=config.BATCH_SIZE
    )
    res = evaluation.evaluate_on(val_ds, model, config.CLASSES)
    print(f"Validation Accuracy: {res['acc']:.4f}")
    print("\nClassification Report:\n", res["report_txt"])

    # 4) Output mirato per modello
    arch, tag, title_add, outdir = _parse_model_tag(model_path)
    _ensure_dir(outdir)

    cm_path = outdir / f"inspect_confusion_matrix_{tag}.png"
    rpt_path = outdir / f"classification_report_{tag}.txt"
    title = f"Confusion Matrix – {arch} ({title_add})"

    evaluation.plot_confusion(res["cm"], config.CLASSES, outpath=str(cm_path), title=title)
    evaluation.save_report(res["report_txt"], str(rpt_path))

    print(f"✅ Confusion matrix saved to: {cm_path}")
    print(f"✅ Classification report saved to: {rpt_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py <path_to_model.keras>")
    else:
        main(sys.argv[1])
