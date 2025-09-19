from rockpaperscissors import data_utils
import tensorflow as tf

def main():
    # 1. Load data
    train_ds, val_ds = data_utils.load_train_val(validation_split=0.2, augment=True)

    # 2. Define a baseline CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Input((128, 128, 3)),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # 3. Train
    history = model.fit(train_ds, validation_data=val_ds, epochs=5)

    # 4. Evaluate
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()
