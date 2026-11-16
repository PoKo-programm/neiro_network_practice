import sys
import matplotlib.pyplot as plt
import numpy as np
from .exceptions import NeuralNetworkError
from .manager import DatasetManager
from .models import NeuralNetwork
from .utils import parse_float, parse_int_list, parse_vector


def print_menu():
    print('''
=== Меню ===
1. Создать сеть
2. Загрузить данные из CSV
3. Обучить сеть
4. Предсказание
5. Сохранить / загрузить веса
6. Показать график ошибки
7. Выход
''')


def create_network():
    raw_layers = input("Размеры слоёв через запятую (например 4,8,1): ")
    activation = input("Активация (relu/sigmoid) [relu]: ") or "relu"
    layers = parse_int_list(raw_layers)
    return NeuralNetwork(layers, activation=activation)


def load_dataset(manager):
    path = input("Путь к CSV-файлу: ").strip()
    manager.load_csv(path)
    method = input("Нормализация (minmax/standard) [minmax]: ") or "minmax"
    manager.normalize(method)
    print(f"Загружено {manager.features.shape[0]} строк, "
          f"{manager.features.shape[1]} признаков")


def train_network(network, manager):
    if network is None or manager.features is None:
        print("Сначала создайте сеть и загрузите данные")
        return
    lr = parse_float(input("Скорость обучения [0.01]: ") or "0.01", "lr")
    epochs = int(parse_float(input("Число эпох [100]: ") or "100", "эпохи"))
    batch = int(parse_float(input("Размер батча [16]: ") or "16", "батч"))
    x_train, x_test, y_train, y_test = manager.split()
    network.train(x_train, y_train, epochs, lr, batch)
    train_pred = network.predict_classes(x_train)
    test_pred = network.predict_classes(x_test)
    train_acc = np.mean(train_pred.flatten() == y_train)
    test_acc = np.mean(test_pred.flatten() == y_test)
    print(f"Точность train: {train_acc:.3f}, test: {test_acc:.3f}")
    print(f"Финальная ошибка: {network.loss_history[-1]:.4f}")


def predict(network):
    if network is None:
        print("Сначала создайте сеть")
        return
    raw = input("Введите признаки через пробел или запятую: ")
    vector = parse_vector(raw)
    probability = network.predict(vector).item()
    label = int(probability >= 0.5)
    print(f"Вероятность класса 1: {probability:.4f}, метка: {label}")


def save_or_load(network):
    if network is None:
        print("Сначала создайте сеть")
        return
    action = input("Действие (save/load): ").strip().lower()
    path = input("Путь к файлу .npz: ").strip()
    if action == "save":
        network.save(path)
        print(f"Веса сохранены в {path}")
    elif action == "load":
        network.load(path)
        print(f"Веса загружены из {path}")
    else:
        print("Неизвестное действие")


def show_loss(network):
    if network is None or not network.loss_history:
        print("История ошибки пуста — сначала обучите сеть")
        return
    plt.plot(network.loss_history)
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка (BCE)")
    plt.title("История обучения")
    plt.grid(True)
    plt.show()


def main():
    network = None
    manager = DatasetManager()

    while True:
        print_menu()
        choice = input("Выбор: ").strip()

        if choice == "7":
            print("Выход")
            return

        try:
            if choice == "1":
                network = create_network()
            elif choice == "2":
                load_dataset(manager)
            elif choice == "3":
                train_network(network, manager)
            elif choice == "4":
                predict(network)
            elif choice == "5":
                save_or_load(network)
            elif choice == "6":
                show_loss(network)
            else:
                print("Нет такого пункта")
        except NeuralNetworkError as exc:
            print(f"Ошибка: {exc}")
        except (ValueError, OSError) as exc:
            print(f"Непредвиденная ошибка: {exc}")


if __name__ == "__main__":
    sys.exit(main())
