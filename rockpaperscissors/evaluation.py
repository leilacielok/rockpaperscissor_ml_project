import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

def evaluate_on(ds, model, class_names, debug=False):
    """
    Evaluate `model` on dataset `ds` (one-hot labels expected) and return metrics.

    Returns dict with:
      - acc (manual, from metrics_from_probs)
      - report_dict (raw sklearn metrics, includes 'accuracy')
      - report_txt (formatted sklearn report)
      - cm, per_class, counts...
      - loss, keras_acc (from model.evaluate)

    """
    y_true = []
    probs_all = []

    for x, y in ds:
        probs = model.predict(x, verbose=0)
        probs_all.append(probs)
        y_true.append(np.argmax(y.numpy(), axis=1))

    y_true = np.concatenate(y_true)
    probs_all = np.concatenate(probs_all, axis=0)

    if debug:
        pmax = probs_all.max(axis=1)
        print("val prob max (mean±std):", float(pmax.mean()), float(pmax.std()))
        print("val prob mean per class:", probs_all.mean(axis=0))

    m = metrics_from_probs(y_true, probs_all, class_names)

    loss, acc = model.evaluate(ds, verbose=0)

    m["loss"] = float(loss)
    m["keras_acc"] = float(acc)  

    return m


def plot_history(history, outdir="reports"):
    Path(outdir).mkdir(exist_ok=True)

    # loss
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    # accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(f"{outdir}/fig_training_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion(
    cm, class_names, outpath="reports/confusion_matrix.png", title="Confusion Matrix"
):
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def save_report(report_txt, outpath="reports/classification_report.txt"):
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(report_txt)

def metrics_from_probs(y_true, probs, class_names):
    """
    Compute classification metrics from ground-truth labels and predicted probabilities.

    Parameters
    ----------
    y_true : array-like, shape (N,)
        Integer class indices (0..K-1).
    probs : np.ndarray, shape (N, K)
        Predicted probabilities (already post-processed if needed).
    class_names : list[str]
        Class names in the same order as columns of probs.

    Returns
    -------
    dict with:
      - y_pred
      - acc
      - macro_f1
      - report_txt
      - cm
      - per_class (dict)
      - true_counts, pred_counts
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1)

    labels = list(range(len(class_names)))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    
    report_txt = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec, rec, f1s, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {
        cls: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1s[i]),
            "support": int(sup[i]),
        }
        for i, cls in enumerate(class_names)
    }

    true_counts = np.bincount(y_true, minlength=len(class_names))
    pred_counts = np.bincount(y_pred, minlength=len(class_names))

    return {
        "y_pred": y_pred,
        "acc": acc,
        "macro_f1": macro_f1,
        "report_txt": report_txt,
        "report_dict": report_dict,
        "cm": cm,
        "per_class": per_class,
        "true_counts": true_counts,
        "pred_counts": pred_counts,
    }
    
def save_misclassified_from_probs(
    file_paths,
    probs,
    y_true,
    class_names,
    top_n=12,
    outpath="reports/misclassified.png",
    pick="confident",
):
    """
    Save a grid of misclassified examples, using precomputed probabilities.

    Parameters
    ----------
    file_paths : list[str]
        List of image file paths aligned with probs / y_true order.
    probs : np.ndarray, shape (N, K)
        Predicted probabilities.
    y_true : array-like, shape (N,)
        Integer class indices.
    class_names : list[str]
        Class names in the same order as probs columns.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)

    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    wrong_idxs = np.where(y_true != y_pred)[0]
    if wrong_idxs.size == 0:
        print("No misclassified image in the split.")
        return

    order = (
        np.argsort(-conf[wrong_idxs])
        if pick == "confident"
        else np.argsort(conf[wrong_idxs])
    )
    sel = wrong_idxs[order][:top_n]

    n = len(sel)
    cols = max(1, int(math.sqrt(n)))
    rows = math.ceil(n / cols)
    plt.figure(figsize=(cols * 2.6, rows * 2.6))

    for k, j in enumerate(sel, 1):
        img = Image.open(file_paths[j]).convert("RGB")
        plt.subplot(rows, cols, k)
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"T:{class_names[y_true[j]]} P:{class_names[y_pred[j]]} c:{conf[j]:.2f}"
        )

    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

def show_misclassified(
    ds,
    model,
    file_paths,
    class_names,
    top_n=12,
    outpath="reports/misclassified.png",
    pick="confident",
):
    """
    Save a grid of misclassified examples.
    NOTE: assumes `file_paths[j]` corresponds to the j-th example produced by `ds`.
    """
    y_true_list, probs_list = [], []

    for x, y in ds:
        probs = model.predict(x, verbose=0)
        probs_list.append(probs)
        y_true_list.append(np.argmax(y.numpy(), axis=1))

    y_true = np.concatenate(y_true_list, axis=0)
    probs_all = np.concatenate(probs_list, axis=0)

    return save_misclassified_from_probs(
        file_paths=file_paths,
        probs=probs_all,
        y_true=y_true,
        class_names=class_names,
        top_n=top_n,
        outpath=outpath,
        pick=pick,
    )

def print_class_histogram(ds, n_classes=3):
    """Return counts per class over the dataset (one-hot labels expected)."""
    counts = None
    for _, y in ds:
        yt = np.argmax(y.numpy(), axis=1)
        bins = np.bincount(yt, minlength=n_classes)
        counts = bins if counts is None else counts + bins
    return counts
