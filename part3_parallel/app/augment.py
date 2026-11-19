import numpy as np


def random_flip(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Случайное горизонтальное отражение."""
    if rng.random() < 0.5:
        return np.fliplr(image)
    return image


def random_rotate(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Случайный поворот на 90/180/270 градусов."""
    k = int(rng.integers(0, 4))
    return np.rot90(image, k)


def add_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.05,
) -> np.ndarray:
    """Добавляет гауссов шум и обрезает значения в [0, 1]."""
    noise = rng.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0.0, 1.0)


def augment_image(image: np.ndarray, seed: int) -> np.ndarray:
    """Применяет цепочку аугментаций. Возвращает новый массив.

    Функция должна быть на верхнем уровне модуля (не lambda, не метод
    класса), чтобы её можно было передать в multiprocessing.Pool —
    встроенный pickle умеет сериализовать только такие функции.
    """
    rng = np.random.default_rng(seed)
    result = random_flip(image, rng)
    result = random_rotate(result, rng)
    result = add_noise(result, rng)
    return result


def augment_pair(args: tuple[np.ndarray, int]) -> np.ndarray:
    """Обёртка для pool.imap: принимает кортеж (image, seed)."""
    image, seed = args
    return augment_image(image, seed)
