from rockpaperscissors import data_utils, config, architectures
import matplotlib.pyplot as plt
import numpy as np, tensorflow as tf, random, os
from sklearn.metrics import classification_report, confusion_matrix
import random, os

tf.random.set_seed(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)
os.environ["PYTHONHASHSEED"] = str(config.SEED)

def main():
    # 1. Load data
    train_ds, val_ds = data_utils.load_train_val(validation_split=0.2, augment=True)

    # 2. Pick a model (baseline for now)
    model = architectures.model_a()
    model.summary()

    # 3. Train with callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)

    # 4. Evaluate on validation set
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")
    best_epoch = 1 + int(np.argmin(history.history["val_loss"]))
    print(f"Early stopped at epoch {best_epoch} (best val_loss={min(history.history['val_loss']):.4f})")

    history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)
    model.save("models/best_baseline.keras")

    # 5. Evaluate on external test set
    try:
        test_ds = data_utils.load_external_test()
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        print(f"External test accuracy: {test_acc:.4f}")

        # 6. Classification report + confusion matrix
        class_names = config.CLASSES
        y_true, y_pred = [], []
        for x, y in test_ds:
            probs = model.predict(x, verbose=0)
            y_pred.append(np.argmax(probs, axis=1))
            y_true.append(np.argmax(y.numpy(), axis=1))
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        print(classification_report(y_true, y_pred, target_names=class_names))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (external test)")
        plt.colorbar()
        ticks = np.arange(len(class_names))
        plt.xticks(ticks, class_names, rotation=45)
        plt.yticks(ticks, class_names)
        plt.xlabel("Predicted");
        plt.ylabel("True")
        plt.tight_layout();
        plt.show()

    except Exception as e:
        print("No external test set found:", e)

    # 7. Plot training curves
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss"); plt.show()
    plt.savefig("reports/fig_training_loss.png", dpi=150, bbox_inches="tight")
    plt.show()

    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.title("Accuracy"); plt.show()
    plt.savefig("reports/fig_training_accuracy.png", dpi=150, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
