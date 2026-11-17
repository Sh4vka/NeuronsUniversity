"""layers module."""


import numpy as np


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.kernel_size = kernel_size

        # He initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size**2))

        self.weights = np.random.randn(
            out_channels, in_channels,
            kernel_size, kernel_size
        ) * scale

        self.biases = np.zeros(out_channels)

    def forward(self, x):
        self.x = x

        # Окна изображения: N, C, H_out, W_out, K, K
        windows = np.lib.stride_tricks.sliding_window_view(
            x,
            (self.kernel_size, self.kernel_size),
            axis=(2, 3)
        )

        self.output = np.einsum(
            "nchwij,ocij->nohw",
            windows,
            self.weights
        ) + self.biases[None, :, None, None]

        return self.output

    def backward(self, grad_output, learning_rate):
        n, c, h, w = self.x.shape
        out_channels, _, k, _ = self.weights.shape

        windows = np.lib.stride_tricks.sliding_window_view(
            self.x,
            (k, k),
            axis=(2, 3)
        )

        grad_weights = np.einsum(
            "nohw,nchwij->ocij",
            grad_output,
            windows
        )

        grad_biases = np.sum(
            grad_output,
            axis=(0, 2, 3)
        )

        grad_x = np.zeros_like(self.x)

        # Распространяем градиент к входным пикселям
        for i in range(k):
            for j in range(k):
                contribution = np.einsum(
                    "nohw,oc->nchw",
                    grad_output,
                    self.weights[:, :, i, j]
                )

                grad_x[:, :, i:i + grad_output.shape[2],
                    j:j + grad_output.shape[3]] += contribution

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_x


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * self.mask


class MaxPool2D:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def forward(self, x):
        self.input_shape = x.shape

        n, c, h, w = x.shape
        p = self.pool_size

        h_out = h // p
        w_out = w // p

        # Отбрасываем остаток, если размер не делится на p
        x = x[:, :, :h_out*p, :w_out*p]

        # N, C, H_out, W_out, p*p
        windows = x.reshape(
            n, c, h_out, p, w_out, p
        ).transpose(0, 1, 2, 4, 3, 5).reshape(
            n, c, h_out, w_out, p*p
        )

        self.max_indices = np.argmax(windows, axis=-1)

        return np.max(windows, axis=-1)

    def backward(self, grad_output):
        n, c, h, w = self.input_shape
        p = self.pool_size

        h_out, w_out = grad_output.shape[2:]

        grad_x = np.zeros((n, c, h, w))

        rows = self.max_indices // p
        cols = self.max_indices % p

        for i in range(p):
            for j in range(p):
                mask = (rows == i) & (cols == j)

                grad_x[
                    :, :, i:h_out*p:p, j:w_out*p:p
                    ] += grad_output * mask

        return grad_x


class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)


class Dense:
    def __init__(self, input_size, output_size):
        scale = np.sqrt(2.0 / input_size)

        self.weights = np.random.randn(
            input_size, output_size
        ) * scale

        self.biases = np.zeros(output_size)

    def forward(self, x):
        self.x = x
        return x @ self.weights + self.biases

    def backward(self, grad_output, learning_rate):
        grad_weights = self.x.T @ grad_output
        grad_biases = np.sum(grad_output, axis=0)

        grad_x = grad_output @ self.weights.T

        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_x


def softmax_cross_entropy(logits, targets):
    # Устойчивая реализация Softmax
    shifted = logits - np.max(logits, axis=1, keepdims=True)

    exp_values = np.exp(shifted)
    probabilities = exp_values / np.sum(
        exp_values, axis=1, keepdims=True
    )

    batch_size = logits.shape[0]

    loss = -np.mean(
        np.log(probabilities[np.arange(batch_size), targets] + 1e-12)
    )

    # Градиент Cross Entropy + Softmax
    grad = probabilities.copy()
    grad[np.arange(batch_size), targets] -= 1.0
    grad /= batch_size

    return loss, probabilities, grad

