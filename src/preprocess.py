import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def main():
    # Load parameters
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    val_size = params["preprocess"]["val_size"]
    seed = params["preprocess"]["seed"]

    # Load raw data
    x_train = np.load("data/raw/x_train.npy")
    y_train = np.load("data/raw/y_train.npy")
    x_test = np.load("data/raw/x_test.npy")
    y_test = np.load("data/raw/y_test.npy")

    # Normalize pixel values from 0-255 to 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Split training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_size,
        random_state=seed,
        stratify=y_train
    )

    # Create processed data directory
    os.makedirs("data/processed", exist_ok=True)

    # Save processed arrays
    np.save("data/processed/x_train.npy", x_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/x_val.npy", x_val)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/x_test.npy", x_test)
    np.save("data/processed/y_test.npy", y_test)

    print("Preprocessing completed successfully - teammate version.")
    print("Training set:", x_train.shape)
    print("Validation set:", x_val.shape)
    print("Test set:", x_test.shape)
    print("Pixel range:", x_train.min(), "to", x_train.max())


if __name__ == "__main__":
    main()