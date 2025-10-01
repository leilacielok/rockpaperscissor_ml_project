import numpy as np, matplotlib.pyplot as plt, math
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from pathlib import Path

def evaluate_on(ds, model, class_names):
    y_true, y_pred = [], []
    for x, y in ds:
        probs = model.predict(x, verbose=0)
        y_pred.append(np.argmax(probs, axis=1))
        y_true.append(np.argmax(y.numpy(), axis=1))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # temporary debug
    print("Unique y_true:", np.unique(y_true), "Unique y_pred:", np.unique(y_pred))

    # temporary mini-check
    probs_all = []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        probs_all.append(p)
    probs_all = np.concatenate(probs_all)
    print("val prob max (mean±std):", probs_all.max(axis=1).mean(), probs_all.max(axis=1).std())
    print("val prob mean per classe:", probs_all.mean(axis=0))

    # explicit labels to avoid bugs
    labels = list(range(len(class_names)))
    report_txt = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    loss, acc = model.evaluate(ds, verbose=0)
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

def plot_confusion(cm, class_names, outpath="reports/confusion_matrix.png", title="Confusion Matrix"):
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
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close()



def save_report(report_txt, outpath="reports/classification_report.txt"):
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    with open(outpath, "w") as f:
        f.write(report_txt)

def show_misclassified(ds, model, file_paths, class_names, top_n=12, outpath="reports/misclassified.png", pick="confident"):
    y_true, y_pred, probs_list = [], [], []
    for x, y in ds:
        probs = model.predict(x, verbose=0)
        probs_list.append(probs)
        y_pred.append(np.argmax(probs, axis=1))
        y_true.append(np.argmax(y.numpy(), axis=1))

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    probs_all = np.concatenate(probs_list, axis=0)
    conf = probs_all.max(axis=1)  # confidence in prediction

    wrong_idxs = np.where(y_true != y_pred)[0]
    if wrong_idxs.size == 0:
        print("No misclassified image in the split.")
        return

    if pick == "confident":
        order = np.argsort(-conf[wrong_idxs])
    else:
        order = np.argsort(conf[wrong_idxs])

    sel = wrong_idxs[order][:top_n]

    n = len(sel)
    cols = max(1, int(math.sqrt(n)))
    rows = math.ceil(n/cols)
    plt.figure(figsize=(cols*2.6, rows*2.6))

    for k, j in enumerate(sel, 1):
        img = Image.open(file_paths[j]).convert("RGB")
        plt.subplot(rows, cols, k); plt.imshow(img); plt.axis("off")
        plt.title(f"T:{class_names[y_true[j]]} P:{class_names[y_pred[j]]} c:{conf[j]: .2f}")
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout(); plt.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close()


# temporary debug function
def print_class_histogram(ds):
    import numpy as np
    counts = None
    for _, y in ds:
        yt = np.argmax(y.numpy(), axis=1)
        bins = np.bincount(yt, minlength=3)
        counts = bins if counts is None else counts + bins
    print("Validation class histogram:", counts)
