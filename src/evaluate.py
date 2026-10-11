import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


def main():
    # Load trained model
    model = tf.keras.models.load_model("models/model.h5")

    # Load processed test data
    x_test = np.load("data/processed/x_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(
        x_test, y_test, verbose=1
    )

    # Generate predictions
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    # Save evaluation metrics
    metrics = {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    # Clothing category labels
    class_names = [
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # Create confusion matrix
    cm = confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close(fig)

    print("\nEvaluation completed successfully!")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("Metrics saved: metrics.json")
    print("Confusion matrix saved: confusion_matrix.png")


if __name__ == "__main__":
    main()