"""main module."""

import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt

import numpy as np

from data import load_mnist
from cnn import CNN


def train_model(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs=3,
    batch_size=32
):
    for epoch in range(epochs):
        # Перемешиваем обучающие примеры
        indices = np.random.permutation(len(x_train))

        x_train = x_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        batches = 0

        for start in range(0, len(x_train), batch_size):
            end = start + batch_size

            images = x_train[start:end]
            labels = y_train[start:end]

            loss, accuracy = model.train_batch(
                images, labels
            )

            epoch_loss += loss
            epoch_accuracy += accuracy
            batches += 1

            if batches % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Batch {batches} | "
                    f"Loss: {loss:.4f} | "
                    f"Accuracy: {accuracy:.4f}"
                )

        train_loss = epoch_loss / batches
        train_accuracy = epoch_accuracy / batches

        test_loss, test_accuracy = model.evaluate(
            x_test, y_test, batch_size
        )

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train loss:     {train_loss:.4f}")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Test loss:      {test_loss:.4f}")
        print(f"Test accuracy:  {test_accuracy:.4f}")
        print("-" * 45)


def save_model(model, path="cnn_weights.npz"):
    np.savez(
        path,
        conv1_weights=model.conv1.weights,
        conv1_biases=model.conv1.biases,
        conv2_weights=model.conv2.weights,
        conv2_biases=model.conv2.biases,
        fc1_weights=model.fc1.weights,
        fc1_biases=model.fc1.biases,
        fc2_weights=model.fc2.weights,
        fc2_biases=model.fc2.biases
    )

    print(f"Model saved: {path}")


def show_prediction(model, image, label):
    probabilities = model.predict(image[np.newaxis, ...])[0]

    predicted = np.argmax(probabilities)

    plt.imshow(image[0], cmap="gray")
    plt.title(
        f"Expected: {label}, Predicted: {predicted}"
    )
    plt.axis("off")
    plt.show()

    print("Class probabilities:")

    for digit, probability in enumerate(probabilities):
        print(f"{digit}: {probability:.4f}")


def main():
    print("Loading MNIST...")

    x_train, y_train, x_test, y_test = load_mnist()

    # Для первого эксперимента используем подмножество
    x_train = x_train[:10000]
    y_train = y_train[:10000]

    x_test = x_test[:2000]
    y_test = y_test[:2000]

    print("Train:", x_train.shape)
    print("Test:", x_test.shape)

    model = CNN(learning_rate=0.01)

    train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=3,
        batch_size=32
    )

    save_model(model)

    print("\nTesting prediction:")

    show_prediction(
        model,
        x_test[0],
        y_test[0]
    )


if __name__ == "__main__":
    main()

