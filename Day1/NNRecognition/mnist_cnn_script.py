import argparse
import math
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D


## Normal run: python mnist_cnn_script.py
## Faster run: python mnist_cnn_script.py --epochs 2 --num-predictions 49

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 1
NUM_CLASSES = 10


def load_and_preprocess_data():
    """
    Loads MNIST and prepares it for a CNN.

    Original shape:
        (num_samples, 28, 28)

    CNN shape:
        (num_samples, 28, 28, 1)
    """
    mnist_dataset = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()

    x_train = x_train.reshape(
        x_train.shape[0],
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
    ).astype("float32")

    x_test = x_test.reshape(
        x_test.shape[0],
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
    ).astype("float32")

    x_train /= 255.0
    x_test /= 255.0

    return x_train, y_train, x_test, y_test


def build_model():
    """
    Builds the CNN model for MNIST digit classification.
    """
    initializer = HeNormal()

    model = Sequential(
        [
            Conv2D(
                8,
                kernel_size=5,
                strides=1,
                activation="relu",
                kernel_initializer=initializer,
                input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
            ),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(
                16,
                kernel_size=5,
                strides=1,
                activation="relu",
                kernel_initializer=initializer,
            ),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=10,
    batch_size=32,
    use_tensorboard=False,
    log_dir=".logs/fit",
):
    """
    Trains the CNN model.
    """
    callbacks = []

    if use_tensorboard:
        log_path = Path(log_dir)

        if log_path.exists():
            shutil.rmtree(log_path)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=str(log_path),
            histogram_freq=1,
        )

        callbacks.append(tensorboard_callback)

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    return history


def plot_training_history(history):
    """
    Plots training and validation loss/accuracy.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Training loss")
    axes[0].plot(history.history["val_loss"], label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss During Training")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Training accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy During Training")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)


def save_and_reload_model(model, model_path):
    """
    Saves and reloads the trained model.
    """
    model.save(model_path)
    loaded_model = tf.keras.models.load_model(model_path)

    return loaded_model


def predict_digits(model, x_test):
    """
    Runs predictions on the test set.
    """
    predictions_one_hot = model.predict(x_test, verbose=0)
    predictions = np.argmax(predictions_one_hot, axis=1)

    return predictions, predictions_one_hot


def plot_prediction_grid(
    x_test,
    y_test,
    predictions,
    numbers_to_display=196,
):
    """
    Displays a grid of predictions.

    Green = correct prediction.
    Red = incorrect prediction.
    """
    numbers_to_display = min(numbers_to_display, len(x_test))
    num_cells = math.ceil(math.sqrt(numbers_to_display))

    plt.figure(figsize=(15, 15))

    for plot_index in range(numbers_to_display):
        predicted_label = predictions[plot_index]
        true_label = y_test[plot_index]

        color_map = "Greens" if predicted_label == true_label else "Reds"

        plt.subplot(num_cells, num_cells, plot_index + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        image = x_test[plot_index].reshape((IMAGE_HEIGHT, IMAGE_WIDTH))

        plt.imshow(image, cmap=color_map)
        plt.xlabel(f"Pred: {predicted_label}\nTrue: {true_label}")

    plt.subplots_adjust(hspace=1.0, wspace=0.5)
    plt.suptitle("MNIST Predictions: Green = Correct, Red = Incorrect", fontsize=16)
    plt.show(block=False)


def plot_confusion_matrix(y_true, y_pred):
    """
    Plots a simple confusion matrix using matplotlib only.
    """
    confusion_matrix = tf.math.confusion_matrix(
        y_true,
        y_pred,
        num_classes=NUM_CLASSES,
    ).numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    threshold = confusion_matrix.max() / 2.0

    for row in range(NUM_CLASSES):
        for col in range(NUM_CLASSES):
            value = confusion_matrix[row, col]
            text_color = "white" if value > threshold else "black"

            plt.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    plt.tight_layout()
    plt.show(block=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-path", type=str, default="digits_recognition_cnn.keras")
    parser.add_argument("--num-predictions", type=int, default=196)
    parser.add_argument("--use-tensorboard", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-summary", action="store_true")

    args = parser.parse_args()

    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    with tf.device("/CPU:0"):
        model = build_model()

    if args.show_summary:
        model.summary()

    history = train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_tensorboard=args.use_tensorboard,
    )

    loaded_model = save_and_reload_model(
        model,
        args.model_path,
    )

    test_loss, test_accuracy = loaded_model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    predictions, _ = predict_digits(
        loaded_model,
        x_test,
    )

    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Saved model to: {args.model_path}")

    if not args.no_plots:
        plot_training_history(history)

        plot_confusion_matrix(
            y_test,
            predictions,
        )

        plot_prediction_grid(
            x_test,
            y_test,
            predictions,
            numbers_to_display=args.num_predictions,
        )

        plt.show()


if __name__ == "__main__":
    main()