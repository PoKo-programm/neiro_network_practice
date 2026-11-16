import numpy as np
from .exceptions import DatasetFileError, MismatchedDataError


class DatasetManager:
    def __init__(self):
        self.features = None
        self.target = None

    def load_csv(self, path):
        try:
            data = np.genfromtxt(path, delimiter=",", skip_header=1)
        except OSError as exc:
            raise DatasetFileError(f"Не удалось открыть {path}: {exc}")
        except ValueError as exc:
            raise DatasetFileError(f"Ошибка разбора CSV: {exc}")
        if data.ndim != 2 or data.shape[1] < 2:
            raise MismatchedDataError(
                "Ожидалась таблица минимум с двумя столбцами"
            )
        if np.isnan(data).any():
            raise MismatchedDataError("В данных есть пропуски (NaN)")
        self.features = data[:, :-1]
        self.target = data[:, -1]
        return self.features, self.target

    def normalize(self, method="minmax"):
        if self.features is None:
            raise MismatchedDataError("Сначала загрузите данные")
        if method == "minmax":
            minimum = self.features.min(axis=0)
            maximum = self.features.max(axis=0)
            span = np.where(maximum - minimum == 0, 1, maximum - minimum)
            self.features = (self.features - minimum) / span
        elif method == "standard":
            mean = self.features.mean(axis=0)
            std = self.features.std(axis=0) + 1e-9
            self.features = (self.features - mean) / std
        else:
            raise MismatchedDataError(f"Неизвестный метод: {method}")

    def split(self, test_ratio=0.2, shuffle=True, seed=42):
        if not 0 < test_ratio < 1:
            raise MismatchedDataError("test_ratio должен быть в (0, 1)")
        n_samples = len(self.features)
        indices = (np.random.default_rng(seed).permutation(n_samples)
                   if shuffle else np.arange(n_samples))
        cut = int(n_samples * (1 - test_ratio))
        train_idx, test_idx = indices[:cut], indices[cut:]
        return (
            self.features[train_idx], self.features[test_idx],
            self.target[train_idx], self.target[test_idx],
        )
