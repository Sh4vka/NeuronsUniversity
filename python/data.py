"""data module."""


import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets


def load_mnist():
    train_data = datasets.MNIST(
        root="../data",
        train=True,
        download=True
    )

    test_data = datasets.MNIST(
        root="../data",
        train=False,
        download=True
    )

    x_train = train_data.data.numpy().astype(np.float32) / 255.0
    y_train = train_data.targets.numpy()

    x_test = test_data.data.numpy().astype(np.float32) / 255.0
    y_test = test_data.targets.numpy()

    # Добавляем размерность канала: N x 1 x 28 x 28
    x_train = x_train[:, np.newaxis, :, :]
    x_test = x_test[:, np.newaxis, :, :]

    return x_train, y_train, x_test, y_test


def show_samples(images, labels, count=10):
    fig, axes = plt.subplots(1, count, figsize=(15, 3))

    for i in range(count):
        axes[i].imshow(images[i, 0], cmap="gray")
        axes[i].set_title(str(labels[i]))
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()

    print("Train:", x_train.shape)
    print("Test:", x_test.shape)

    show_samples(x_train, y_train)

