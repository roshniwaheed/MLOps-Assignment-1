import os
import csv
import numpy as np
import yaml
import tensorflow as tf


def main():
    # Load training parameters
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)["train"]

    dense_units = params["dense_units"]
    dropout_rate = params["dropout_rate"]
    learning_rate = params["learning_rate"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]

    # Load processed training and validation data
    x_train = np.load("data/processed/x_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    x_val = np.load("data/processed/x_val.npy")
    y_val = np.load("data/processed/y_val.npy")

    # Build fully connected ANN
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Display model architecture
    model.summary()

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    # Create output directory
    os.makedirs("models", exist_ok=True)

    # Save trained model
    model.save("models/model.h5")

    # Save training history
    with open("models/history.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "epoch", "loss", "accuracy", "val_loss", "val_accuracy"
        ])

        for i in range(len(history.history["loss"])):
            writer.writerow([
                i + 1,
                history.history["loss"][i],
                history.history["accuracy"][i],
                history.history["val_loss"][i],
                history.history["val_accuracy"][i]
            ])

    print("\nTraining completed successfully!")
    print("Model saved: models/model.h5")
    print("History saved: models/history.csv")


if __name__ == "__main__":
    main()