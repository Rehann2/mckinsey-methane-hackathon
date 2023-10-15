from utils.dataloader import LoadData
import argparse
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split


def define_model():
    """Define the keras sequential model

    Returns:
        tf.keras.model: The model to train
    """
    model = tf.keras.Sequential(
        [
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            # Fully connected layers
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC()]
    )
    return model


def main(train_args):
    """Train the model and save it in models directory

    Args:
        train_args (argparse.ArgumentParser): Dictionary for terminal arguments
    """
    # Load and preprocess your data
    data = LoadData(
        metadata_path=train_args.metadata_path,
        image_data_path=train_args.image_data_path,
    )

    X, y = data.get_train_data()
    X = data.normalize_data(X)
    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_args.val_size, random_state=42
    )
    if train_args.augment:
        X_train, y_train = data.augment_data(
            X_train, y_train, batch_size=train_args.aug_batch
        )
    # Create and compile your Keras model
    model = define_model()

    # Train and evaluate the model
    model.fit(
        X_train,
        y_train,
        epochs=train_args.epochs,
        batch_size=train_args.batch_size,
        validation_data=(X_test, y_test),
    )

    model.save(f"{train_args.model_path}{train_args.model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model from the Terminal")
    parser.add_argument("model_name", help="name the model")
    parser.add_argument(
        "--metadata_path", default="data/dataset/train_data/metadata.csv"
    )
    parser.add_argument("--image_data_path", default="data/dataset/train_data/")
    parser.add_argument(
        "--model_path", default="model/", help="directory where models are stored"
    )
    parser.add_argument("--val_size", help="test split ratio ", default=0.2)
    parser.add_argument("--epochs", help="number of epochs in the model", default=10)
    parser.add_argument(
        "--batch_size", help="batch size used in model training", default=32
    )
    parser.add_argument(
        "--group_split",
        help="Specifying this argument leads to group splitting",
        action="store_true",
    )
    parser.add_argument(
        "--augment", action="store_true", help="Whether to augment data or not"
    )
    parser.add_argument(
        "--aug_batch", help="batch size of data augmentation", default=32
    )
    args = parser.parse_args()
    main(args)
