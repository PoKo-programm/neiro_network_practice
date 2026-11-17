import numpy as np
from .exceptions import InvalidLayerSizeError, MismatchedDataError


class NeuralNetwork:
    """Полносвязная сеть с произвольным числом слоёв.

    :param layer_sizes: размеры слоёв, например [n_features, 16, 1]
    :param activation: 'relu' или 'sigmoid'
    :param seed: зерно генератора случайных чисел
    """
    SUPPORTED_ACTIVATIONS = ("relu", "sigmoid")

    def __init__(self, layer_sizes, activation="relu", seed=None):
        if len(layer_sizes) < 2:
            raise InvalidLayerSizeError("Нужно минимум два слоя")
        if any(size <= 0 for size in layer_sizes):
            raise InvalidLayerSizeError("Размеры слоёв должны быть больше 0")
        if activation not in self.SUPPORTED_ACTIVATIONS:
            raise InvalidLayerSizeError(
                f"Активация '{activation}' не поддерживается"
            )
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.rng = np.random.default_rng(seed)
        self.weights, self.biases = self._init_params()
        self.loss_history = []

    def _init_params(self):
        """Инициализация весов (Xavier для sigmoid, He для ReLU)."""
        weights, biases = [], []
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            if self.activation_name == "relu":
                scale = np.sqrt(2.0 / n_in)
            else:
                scale = np.sqrt(1.0 / n_in)
            weights.append(self.rng.normal(0, scale, (n_in, n_out)))
            biases.append(np.zeros((1, n_out)))
        return weights, biases

    def _activation(self, z):
        if self.activation_name == "relu":
            return np.maximum(0, z)
        return 1.0 / (1.0 + np.exp(-z))

    def _activation_derivative(self, z):
        if self.activation_name == "relu":
            return (z > 0).astype(float)
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        return sigmoid * (1 - sigmoid)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):
        """Прямой проход. Возвращает (выход, кэш для backprop)."""
        if X.ndim != 2:
            raise MismatchedDataError("X должен быть двумерным массивом")
        if X.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                f"Ожидалось {self.layer_sizes[0]} признаков, "
                f"получено {X.shape[1]}"
            )
        activations = [X]
        zs = []
        current = X
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = current @ weight + bias
            zs.append(z)
            if i < len(self.weights) - 1:
                current = self._activation(z)
            else:
                current = self._sigmoid(z)
            activations.append(current)
        return current, (activations, zs)

    def backward(self, cache, y_true, learning_rate):
        """Обратный проход и шаг градиентного спуска."""
        activations, zs = cache
        batch_size = y_true.shape[0]
        delta = (activations[-1] - y_true) / batch_size
        for i in reversed(range(len(self.weights))):
            d_weights = activations[i].T @ delta
            d_biases = np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * \
                    self._activation_derivative(zs[i - 1])
            self.weights[i] -= learning_rate * d_weights
            self.biases[i] -= learning_rate * d_biases

    def train(self, X, y, epochs, learning_rate, batch_size):
        """Обучение мини-батчами с записью истории ошибки."""
        y = np.asarray(y).reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise MismatchedDataError("X и y разной длины")
        for _ in range(epochs):
            indices = self.rng.permutation(X.shape[0])
            epoch_loss = 0.0
            for start in range(0, X.shape[0], batch_size):
                batch_idx = indices[start:start + batch_size]
                x_batch, y_batch = X[batch_idx], y[batch_idx]
                output, cache = self.forward(x_batch)
                loss = -np.mean(
                    y_batch * np.log(output + 1e-9)
                    + (1 - y_batch) * np.log(1 - output + 1e-9)
                )
                epoch_loss += loss * len(batch_idx)
                self.backward(cache, y_batch, learning_rate)
            self.loss_history.append(epoch_loss / X.shape[0])

    def predict(self, X):
        """Возвращает вероятности принадлежности к классу 1."""
        output, _ = self.forward(X)
        return output

    def predict_classes(self, X, threshold=0.5):
        """Возвращает бинарные метки (0/1)."""
        return (self.predict(X) >= threshold).astype(int)

    def save(self, path):
        """Сохраняет веса и конфигурацию в .npz."""
        data = {f"weight_{i}": w for i, w in enumerate(self.weights)}
        data.update({f"bias_{i}": b for i, b in enumerate(self.biases)})
        np.savez(path, **data)

    def load(self, path):
        """Загружает веса из .npz, проверяя размерности."""
        with np.load(path) as data:
            n_layers = len(self.weights)
            try:
                weights = [data[f"weight_{i}"] for i in range(n_layers)]
                biases = [data[f"bias_{i}"] for i in range(n_layers)]
            except KeyError as exc:
                raise MismatchedDataError(
                    f"В файле нет нужного ключа: {exc}"
                )
            for i, w in enumerate(weights):
                if w.shape != self.weights[i].shape:
                    raise MismatchedDataError(
                        f"Слой {i}: ожидалось {self.weights[i].shape}, "
                        f"получено {w.shape}"
                    )
            self.weights, self.biases = weights, biases
