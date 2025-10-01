from pathlib import Path
import shutil

reports = Path("reports")
models = [d.name for d in reports.iterdir() if d.is_dir() and d.name.startswith("model_")]
if not models:
    models = ["model_a","model_b","model_c","model_d"]  # fallback

def move(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"→ {src.name}  ->  {dst}")
        shutil.move(str(src), str(dst))

# Sposta i file con prefisso model_* nelle rispettive cartelle
for m in models:
    move(reports / f"{m}_val_confusion_matrix.png",      reports / m / "val_confusion_matrix.png")
    move(reports / f"{m}_val_misclassified.png",         reports / m / "val_misclassified.png")
    move(reports / f"{m}_val_classification_report.txt", reports / m / "val_classification_report.txt")
    move(reports / f"{m}_test_confusion_matrix.png",     reports / m / "test_confusion_matrix.png")
    move(reports / f"{m}_test_classification_report.txt",reports / m / "test_classification_report.txt")

# Metti i file “generici” che non sappiamo a quale modello appartengano in una cartella di archivio
legacy = reports / "_legacy"
for fname in ["confusion_matrix.png", "fig_confusion_matrix.png",
              "fig_training_accuracy.png", "fig_training_loss.png"]:
    move(reports / fname, legacy / fname)

print("Done.")
