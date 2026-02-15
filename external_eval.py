from pathlib import Path

from keras.models import load_model

from rockpaperscissors import config, data_utils, evaluation

# 1. Final model
MODEL_PATH = "models/final_best.keras"
model = load_model(MODEL_PATH)

# 2. External dataset
test_ds = data_utils.load_external_test()

# 3. Evaluation
results = evaluation.evaluate_on(test_ds, model, config.CLASSES)

# 4. Output directory
outdir = Path("reports/model_c_final")
outdir.mkdir(parents=True, exist_ok=True)

# 5. Save classification report
evaluation.save_report(
    results["report_txt"], str(outdir / "test_classification_report.txt")
)

# 6. Save confusion matrix
evaluation.plot_confusion(
    results["cm"],
    config.CLASSES,
    outpath=str(outdir / "test_confusion_matrix.png"),
    title="Confusion Matrix (external test – rps-cv-images)",
)

print("External evaluation completed successfully.")
