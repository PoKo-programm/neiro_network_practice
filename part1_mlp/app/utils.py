import numpy as np
from .exceptions import MismatchedDataError


def parse_int_list(raw):
    """Преобразует '4,8,1' в [4, 8, 1] с проверкой."""
    try:
        values = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise MismatchedDataError(f"Ожидались целые числа: {exc}")
    if not values:
        raise MismatchedDataError("Список размеров пуст")
    return values


def parse_float(raw, name="значение"):
    """Безопасно приводит строку к float."""
    try:
        return float(raw)
    except ValueError as exc:
        raise MismatchedDataError(f"{name}: ожидалось число ({exc})")


def parse_vector(raw):
    """Преобразует '1 2 3' или '1,2,3' в numpy-вектор строки."""
    tokens = raw.replace(",", " ").split()
    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise MismatchedDataError(f"Некорректный вектор: {exc}")
    return np.array(values, dtype=float).reshape(1, -1)
