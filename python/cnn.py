"""cnn module."""


import numpy as np

from layers import (
Conv2D,
ReLU,
MaxPool2D,
Flatten,
Dense,
softmax_cross_entropy
)


class CNN:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

        # Сверточные слои
        self.conv1 = Conv2D(1, 8, 3)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2)

        self.conv2 = Conv2D(8, 16, 3)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2)

        # Классификатор
        self.flatten = Flatten()
        self.fc1 = Dense(400, 64)
        self.relu3 = ReLU()
        self.fc2 = Dense(64, 10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.fc2.forward(x)

        return x

    def backward(self, grad):
        grad = self.fc2.backward(
            grad, self.learning_rate
        )

        grad = self.relu3.backward(grad)

        grad = self.fc1.backward(
            grad, self.learning_rate
        )

        grad = self.flatten.backward(grad)

        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)

        grad = self.conv2.backward(
            grad, self.learning_rate
        )

        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)

        grad = self.conv1.backward(
            grad, self.learning_rate
        )

        return grad

    def train_batch(self, images, labels):
        logits = self.forward(images)

        loss, probabilities, grad = softmax_cross_entropy(
            logits, labels
        )

        self.backward(grad)

        predictions = np.argmax(probabilities, axis=1)
        accuracy = np.mean(predictions == labels)

        return loss, accuracy

    def predict(self, images):
        logits = self.forward(images)

        shifted = logits - np.max(
            logits, axis=1, keepdims=True
        )

        exp_values = np.exp(shifted)

        probabilities = exp_values / np.sum(
            exp_values, axis=1, keepdims=True
        )

        return probabilities

    def evaluate(self, images, labels, batch_size=32):
        total_correct = 0
        total_loss = 0.0

        for start in range(0, len(images), batch_size):
            end = start + batch_size

            batch_images = images[start:end]
            batch_labels = labels[start:end]

            logits = self.forward(batch_images)

            loss, probabilities, _ = softmax_cross_entropy(
                logits, batch_labels
            )

            predictions = np.argmax(probabilities, axis=1)

            total_correct += np.sum(
                predictions == batch_labels
            )

            total_loss += loss * len(batch_images)

        accuracy = total_correct / len(images)
        average_loss = total_loss / len(images)

        return average_loss, accuracy

