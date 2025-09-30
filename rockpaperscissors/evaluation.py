from pathlib import Path
import numpy as np, matplotlib.pyplot as plt, math
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

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


def save_report(report_txt, outpath="reports/classification_report.txt"):
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    with open(outpath, "w") as f:
        f.write(report_txt)

def show_misclassified(ds, model, file_paths, class_names, top_n=12, outpath="reports/misclassified.png"):
    y_true, y_pred, idxs = [], [], []
    i0 = 0
    for x, y in ds:
        probs = model.predict(x, verbose=0)
        yp = np.argmax(probs, axis=1)
        yt = np.argmax(y.numpy(), axis=1)
        n = len(yt)
        y_true.append(yt); y_pred.append(yp)
        idxs.extend(range(i0, i0+n))
        i0 += n
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)

    wrong = np.where(y_true != y_pred)[0]
    if wrong.size == 0:
        print("Nessuna immagine misclassificata nella split fornita.")
        return
    wrong = wrong[:top_n]

    cols = int(math.sqrt(top_n)); rows = math.ceil(top_n/cols)
    plt.figure(figsize=(cols*2.6, rows*2.6))
    for k, j in enumerate(wrong, 1):
        img = Image.open(file_paths[j]).convert("RGB")
        plt.subplot(rows, cols, k); plt.imshow(img); plt.axis("off")
        plt.title(f"T:{class_names[y_true[j]]} P:{class_names[y_pred[j]]}")
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout(); plt.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close()