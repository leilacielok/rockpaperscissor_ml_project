import os, math, argparse, itertools
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix
from rockpaperscissors import config, evaluation

AUTOTUNE = tf.data.AUTOTUNE

# ------------------------- util -------------------------
def _ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def _artifact_tag(model_path: str) -> str:
    return Path(model_path).stem

def _save_summary(
    outpath, classes, acc, macro_f1, per_class, true_counts, pred_counts,
    settings, used_perm, pred_prior=None, target_prior=None, weights=None
):
    """
    overall: accuracy, macro f1
    per_class: dict {class_name: {precision, recall, f1, support}}
    settings: dict with flag/parameters used (resize, TTA, zero-bias, recalib...)
    """
    lines = []
    lines.append("# Evaluation Summary")
    lines.append("")
    lines.append("## Overall")
    lines.append(f"- Accuracy: {acc:.4f}")
    lines.append(f"- Macro F1: {macro_f1:.4f}")
    lines.append("")
    lines.append("## Per-class metrics")
    for c in classes:
        m = per_class[c]
        lines.append(f"- {c}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  (support={m['support']})")
    lines.append("")
    lines.append("## Distributions")
    lines.append(f"- True counts   : {true_counts.tolist()}")
    lines.append(f"- Pred counts   : {pred_counts.tolist()}")
    lines.append("")
    lines.append("## Settings")
    for k, v in settings.items():
        lines.append(f"- {k}: {v}")
    if used_perm is not None:
        lines.append(f"- Class permutation used: {used_perm}")
    if pred_prior is not None:
        lines.append(f"- Pred prior (mean probs pre-recalib): {np.round(pred_prior, 3)}")
    if target_prior is not None:
        lines.append(f"- Target prior: {np.round(target_prior, 3)}")
    if weights is not None:
        lines.append(f"- Recalib weights: {np.round(weights, 3)}")
    lines.append("")

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ------------------------- loader (EXIF + resize) -------------------------
def _decode_image_pil(path):
    p = path.numpy().decode()
    img = Image.open(p)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB") # (R,B,G)
    return np.array(img)  # image pixels matrix HxWx3

def _load_img(path, img_size, resize_mode="pad"):
    x = tf.py_function(_decode_image_pil, [path], Tout=tf.uint8)
    x.set_shape([None, None, 3])
    x = tf.image.convert_image_dtype(x, tf.float32)   # [0,1]
    if resize_mode == "pad":
        x = tf.image.resize_with_pad(x, img_size, img_size, method="bilinear", antialias=True)
    else:
        x = tf.image.resize(x, (img_size, img_size), method="bilinear", antialias=True)
    return x

def _parse(path, label, img_size, n_classes, resize_mode="pad"):
    return _load_img(path, img_size, resize_mode=resize_mode), tf.one_hot(label, n_classes)

def load_labeled_dir(root_dir, batch_size=None, resize_mode="pad"):
    """
    Load a labeled image dataset from a directory structured by class names.

    The function expects `root_dir` to contain one subdirectory per class,
    with subdirectory names matching `config.CLASSES` (e.g. `rock/`, `paper/`,
    `scissors/`). Each image file found in these subdirectories is assigned
    a label corresponding to the index of its class in `config.CLASSES`.

    Images are loaded using PIL with EXIF orientation correction, converted
    to RGB to guarantee three channels, normalized to float32 in [0, 1],
    and resized to a fixed square shape according to `resize_mode`.

    The resulting dataset yields batches of `(image, label)` pairs, where
    images have shape `(IMG_SIZE, IMG_SIZE, 3)` and labels are one-hot
    encoded vectors of length `n_classes`.

    Returns
    ds as tf.data.Dataset: TensorFlow dataset yielding batches of `(image, label)` pairs,
    file_paths as list of str: list of absolute file paths corresponding to the images in the
        dataset, in the same order as the dataset elements. 

    RuntimeError if no image files are found in the expected class subdirectories (no out-of-distribution evaluation).
    """
    bs = batch_size or config.BATCH_SIZE
    class_names = list(config.CLASSES)
    n_classes = len(class_names)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    # label each file with class index
    files, labels = [], []
    for c in class_names:
        cdir = Path(root_dir) / c
        for p in sorted(cdir.glob("*")):
            if p.is_file():
                files.append(str(p))
                labels.append(class_to_idx[c])

    if not files:
        raise RuntimeError(f"No image found in {root_dir} in subfolders {class_names}")

    print("Using class order:", class_names, f" | Found: {len(files)} files")
    files_tf = tf.constant(files)
    labels_tf = tf.constant(labels, dtype=tf.int32)

    img_size = getattr(config, "IMG_SIZE", 96)
    ds = tf.data.Dataset.from_tensor_slices((files_tf, labels_tf)) \
        .map(lambda p, y: _parse(p, y, img_size, n_classes, resize_mode), # map: decode+resize+one-hot
             num_parallel_calls=AUTOTUNE) \
        .batch(bs).cache().prefetch(AUTOTUNE)
    return ds, files

# ------------------------- model introspection -------------------------
def _model_has_rescale_1_over_255(model) -> bool:
    for lyr in model.layers:
        if isinstance(lyr, tf.keras.layers.InputLayer):
            continue
        if isinstance(lyr, tf.keras.layers.Rescaling):
            scale = getattr(lyr, "scale", None)
            offset = getattr(lyr, "offset", 0.0)
            if scale is not None and abs(float(scale) - 1.0/255.0) < 1e-6 and abs(float(offset)) < 1e-6:
                return True
        break
    return False

def _get_last_dense_bias(model):
    for lyr in reversed(model.layers):
        if isinstance(lyr, tf.keras.layers.Dense):
            w = lyr.get_weights()
            if len(w) == 2:  # [W, b]
                return w[1] # b = [b_rick, b_paper, b_scissors]
            break
    return None

# remove learned prior: p' ∝ p * exp(-b)
def _apply_zero_bias_to_probs(model, probs):
    b = _get_last_dense_bias(model) # b = [b_rick, b_paper, b_scissors]
    if b is None:
        return probs
    w = np.exp(-b).reshape(1, -1) # weights vector where w_k = exp(-b_k)
    p = probs * w # scale probs: p'_k = p_k * w_k
    p = p / p.sum(axis=1, keepdims=True) # normalize again
    return p

# ------------------------- TTA -------------------------
def _tta_views(xb, use_rot=False):
    img_size = getattr(config, "IMG_SIZE", 96)
    def _resize_warp(x): 
        return tf.image.resize(x, (img_size, img_size), method="bilinear", antialias=True)

    views = [xb, tf.image.flip_left_right(xb)] # views with different augmentations
    for frac in (0.9, 0.8):
        crop = _resize_warp(tf.image.central_crop(xb, frac))
        views += [crop, tf.image.flip_left_right(crop)]

    if use_rot:
        try:
            import tensorflow_addons as tfa
            for deg in (-12, +12):
                views.append(tfa.image.rotate(xb, np.deg2rad(deg), interpolation='bilinear'))
        except Exception:
            pass
    return views

def _predict_with_tta(model, xb, use_rot=False, pre_scale_255=False):
    views = _tta_views(xb, use_rot=use_rot)
    if pre_scale_255:
        views = [tf.clip_by_value(v * 255.0, 0.0, 255.0) for v in views]
    preds = [model(vi, training=False).numpy() for vi in views] # preds for each view
    return np.mean(preds, axis=0)

# ------------------------- post-proc -------------------------
def _best_perm_from_probs(y_true, probs):
    k = probs.shape[1]
    best_acc, best_perm = -1.0, tuple(range(k))
    for perm in itertools.permutations(range(k), k):
        acc = (probs[:, list(perm)].argmax(axis=1) == y_true).mean()
        if acc > best_acc:
            best_acc, best_perm = acc, perm
    return best_perm, float(best_acc)

def _recalibrate_probs(probs, y_true=None, mode="uniform", alpha=1.0, eps=1e-6):
    K = probs.shape[1]
    pred_prior = probs.mean(axis=0)
    if mode == "empirical" and y_true is not None:
        counts = np.bincount(y_true, minlength=K).astype(np.float64)
        target_prior = counts / max(1, counts.sum())
    else:
        target_prior = np.ones(K, dtype=np.float64) / K
    w = np.power((target_prior + eps) / (pred_prior + eps), alpha)
    probs_corr = probs * w.reshape(1, K)
    probs_corr = probs_corr / probs_corr.sum(axis=1, keepdims=True)
    return probs_corr, w, target_prior, pred_prior

# ------------------------- viz -------------------------
def _save_misclassified_from_probs(file_paths, probs, y_true, class_names,
                                   top_n=12, outpath="reports/misclassified.png",
                                   pick="confident"):
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    wrong_idxs = np.where(y_true != y_pred)[0]
    if wrong_idxs.size == 0:
        print("No misclassified image in the split.")
        return
    order = np.argsort(-conf[wrong_idxs]) if pick == "confident" else np.argsort(conf[wrong_idxs])
    sel = wrong_idxs[order][:top_n]

    n = len(sel) 
    cols = max(1, int(math.sqrt(n)))
    rows = math.ceil(n/cols)
    plt.figure(figsize=(cols*2.6, rows*2.6))
    for k, j in enumerate(sel, 1):
        img = Image.open(file_paths[j]).convert("RGB")
        plt.subplot(rows, cols, k); plt.imshow(img); plt.axis("off")
        plt.title(f"T:{class_names[y_true[j]]}  P:{class_names[y_pred[j]]}  c:{conf[j]:.2f}")
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout(); plt.savefig(outpath, dpi=150, bbox_inches="tight"); plt.close()

# ------------------------- main -------------------------
def main(model_path: str, data_dir: str, outdir: str | None):
    USE_TTA = getattr(config, "EVAL_TTA", True)
    USE_TTA_ROT = USE_TTA and getattr(config, "EVAL_TTA_ROT", False)
    RESIZE_MODE = getattr(config, "EVAL_RESIZE_MODE", "pad")   # 'pad' | 'warp'
    RECALIB_MODE = getattr(config, "EVAL_RECALIB", "uniform")  # 'off' | 'uniform' | 'empirical'
    RECALIB_ALPHA = getattr(config, "RECALIB_ALPHA", 1.0)
    ZERO_BIAS = getattr(config, "EVAL_ZERO_BIAS", False)

    tag = _artifact_tag(model_path)

    # -------- outdir auto / opzionale --------
    outroot = getattr(config, "EVAL_OUTROOT", "reports/custom_eval_myhands")
    prefix  = getattr(config, "EVAL_OUTDIR_PREFIX", "my_hands_")
    always_sub = getattr(config, "EVAL_ALWAYS_SUBDIR", False)

    if outdir is None:
        outdir = os.path.join(outroot, f"{prefix}{tag}")
    elif always_sub:
        outdir = os.path.join(outdir, f"{prefix}{tag}")

    print(f"🔍 Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    pre_scale_255 = _model_has_rescale_1_over_255(model)
    if pre_scale_255:
        print("⚠️  Detected model Rescaling(1/255) at input → multiply inputs ×255 before prediction.")

    print(f"📁 Evaluating on labeled folder: {data_dir}")
    ds, file_paths = load_labeled_dir(data_dir, resize_mode=RESIZE_MODE)

    probs_all, ytrue_all = [], []
    for xb, yb in ds:
        if USE_TTA:
            p = _predict_with_tta(model, xb, use_rot=USE_TTA_ROT, pre_scale_255=pre_scale_255)
        else:
            x = tf.clip_by_value(xb*255.0, 0.0, 255.0) if pre_scale_255 else xb
            p = model.predict(x, verbose=0)
        probs_all.append(p)
        ytrue_all.append(yb.numpy())

    probs = np.concatenate(probs_all, axis=0)
    y_true = np.argmax(np.concatenate(ytrue_all, axis=0), axis=1)

    # zero-bias
    if ZERO_BIAS:
        probs = _apply_zero_bias_to_probs(model, probs)
        print("▶ Applied zero-bias to final Dense (removed learned class prior).")

    # auto-permutazione classi
    identity_pred = probs.argmax(axis=1)
    id_acc = (identity_pred == y_true).mean()
    best_perm, best_acc = _best_perm_from_probs(y_true, probs)
    used_perm = None
    if best_acc > id_acc + 1e-6:
        print(f"🔁 [auto-fix] Using class permutation {best_perm} (acc={best_acc:.3f}) "
              f"instead of identity (acc={id_acc:.3f}) to align outputs to {config.CLASSES}.")
        probs = probs[:, list(best_perm)]
        used_perm = best_perm
    else:
        print(f"✅ Class order appears aligned (identity perm). acc={id_acc:.3f}")

    # ricalibrazione
    pred_prior_used = None
    target_prior_used = None
    weights_used = None

    if RECALIB_MODE == "uniform":
        probs, w, target_prior, pred_prior = _recalibrate_probs(probs, y_true, "uniform", RECALIB_ALPHA)
        pred_prior_used, target_prior_used, weights_used = pred_prior, target_prior, w
        print(f"▶ Using UNIFORM recalibration (alpha={RECALIB_ALPHA}). "
              f"pred_prior={np.round(pred_prior, 3)} weights={np.round(w, 2)}")
    elif RECALIB_MODE == "empirical":
        probs, w, target_prior, pred_prior = _recalibrate_probs(probs, y_true, "empirical", RECALIB_ALPHA)
        pred_prior_used, target_prior_used, weights_used = pred_prior, target_prior, w
        print(f"▶ Using EMPIRICAL recalibration (alpha={RECALIB_ALPHA}). "
              f"target_prior={np.round(target_prior, 3)} weights={np.round(w, 2)}")
    else:
        print("▶ Using RAW posterior (no recalibration).")

    # metriche finali
    y_pred = probs.argmax(axis=1)
    labels = list(range(len(config.CLASSES)))
    report_txt = classification_report(y_true, y_pred, labels=labels, target_names=config.CLASSES, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float((y_true == y_pred).mean())

    print(("\nAccuracy (TTA): " if USE_TTA else "\nAccuracy: ") + f"{acc:.4f}\n")
    print(report_txt)

    # ---- riepilogo numerico ----
    true_counts = np.bincount(y_true, minlength=len(config.CLASSES))
    pred_counts = np.bincount(y_pred, minlength=len(config.CLASSES))
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec, rec, f1s, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    per_class = {
        cls: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1s[i]), "support": int(sup[i])}
        for i, cls in enumerate(config.CLASSES)
    }

    settings = {
        "IMG_SIZE": getattr(config, "IMG_SIZE", 96),
        "RESIZE_MODE": RESIZE_MODE,
        "USE_TTA": USE_TTA,
        "EVAL_TTA_ROT": USE_TTA_ROT,
        "EVAL_ZERO_BIAS": ZERO_BIAS,
        "EVAL_RECALIB": RECALIB_MODE,
        "RECALIB_ALPHA": RECALIB_ALPHA,
    }

    summary_path = os.path.join(outdir, f"summary_{tag}.txt")
    _save_summary(
        summary_path, config.CLASSES, acc, macro_f1, per_class, true_counts, pred_counts,
        settings, used_perm, pred_prior=pred_prior_used, target_prior=target_prior_used, weights=weights_used
    )
    print(f"📝  Saved summary to {summary_path}")

    # salvataggi
    _ensure_dir(outdir)
    evaluation.save_report(report_txt, os.path.join(outdir, f"classification_report_{tag}.txt"))
    evaluation.plot_confusion(cm, config.CLASSES,
                              os.path.join(outdir, f"confusion_matrix_{tag}.png"),
                              title=f"Confusion Matrix – {tag.replace('_', ' ')}")
    try:
        _save_misclassified_from_probs(
            file_paths, probs, y_true, config.CLASSES,
            top_n=12, outpath=os.path.join(outdir, f"misclassified_{tag}.png")
        )
        print(f"🖼️  Saved misclassified grid to {os.path.join(outdir, f'misclassified_{tag}.png')}")
    except Exception as e:
        print(f"(info) Unable to save misclassified grid: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved .keras model on your labeled hand-gesture photos.")
    parser.add_argument("--model", required=True, help="Path to .keras model (e.g., models/model_a_best.keras)")
    parser.add_argument("--dir", default="my_hands", help="Root folder with subfolders rock/paper/scissors")
    parser.add_argument(
        "--outdir",
        default=None,
        help=("Where to save reports/plots. If omitted, uses "
              "<EVAL_OUTROOT>/<EVAL_OUTDIR_PREFIX><model_tag>. "
              "If EVAL_ALWAYS_SUBDIR=True and this is provided, "
              "it creates a subfolder <prefix><tag> inside it.")
    )
    args = parser.parse_args()
    main(args.model, args.dir, args.outdir)

