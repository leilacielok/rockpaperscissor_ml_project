from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_on(ds, model, class_names):
    loss, acc = model.evaluate(ds, verbose=0)

    y_true, y_pred = [], []
    for x, y in ds:
        probs = model.predict(x, verbose=0)
        y_pred.append(np.argmax(probs, axis=1))
        y_true.append(np.argmax(y.numpy(), axis=1))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    report_txt = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)
    return {"loss": loss, "acc": acc, "report_txt": report_txt, "cm": cm}

def plot_history(history, outdir="reports"):
    Path(outdir).mkdir(exist_ok=True)

    # loss
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")
    plt.tight_layout(); plt.savefig(f"{outdir}/fig_training_loss.png", dpi=150, bbox_inches="tight"); plt.close()

    # accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy")
    plt.tight_layout(); plt.savefig(f"{outdir}/fig_training_accuracy.png", dpi=150, bbox_inches="tight"); plt.close()

def plot_confusion(cm, class_names, outpath="reports/confusion_matrix.png"):
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (external test)")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)
    # numbers on cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close()
