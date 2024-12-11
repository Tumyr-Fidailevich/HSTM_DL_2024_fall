import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

print("GPU: ", torch.cuda.is_available())


class MLP(nn.Module):
    """
    Класс дл описания многоуровневого перцептрона
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# def add_acceleration(data, dt=1):
#     """
#     Добавляет ускорения для каждого тела в массив данных.
#     :param data: numpy array, исходный массив данных
#     :param dt: интервал времени для вычисления ускорений (по умолчанию 1)
#     :return: новый массив данных с добавленными ускорениями
#     """
#     num_bodies_in_years = data.shape[1] // 4  # Каждое тело имеет 4 столбца: x, y, vx, vy
#
#     # Заполняем первый год нулями - данных нет
#     new_data = np.zeros((data.shape[0], num_bodies_in_years * 6))  # 6 столбцов: x, y, vx, vy, ax, ay
#
#     # Заполнение нового массива для каждого года, начиная со второго
#     for i in range(1, data.shape[0]):  # Начинаем со второй строки (второго года)
#         row = []
#         for body in range(num_bodies_in_years):
#             # Индексы текущего тела
#             idx_x = body * 4
#             idx_y = body * 4 + 1
#             idx_vx = body * 4 + 2
#             idx_vy = body * 4 + 3
#
#             # Текущие координаты и скорости
#             x, y = data[i, idx_x], data[i, idx_y]
#             vx, vy = data[i, idx_vx], data[i, idx_vy]
#
#             # Предыдущие скорости
#             prev_vx = data[i-1, idx_vx]
#             prev_vy = data[i-1, idx_vy]
#
#             # Вычисляем ускорения
#             ax = (vx - prev_vx) / dt
#             ay = (vy - prev_vy) / dt
#
#             # Добавляем координаты, скорости и ускорения для текущего тела
#             row.extend([x, y, vx, vy, ax, ay])
#
#         new_data[i] = row  # Заполняем новую строку
#
#     return new_data

def add_acceleration(data, dt=1):
    """
    Добавляет ускорения для каждого тела в массив данных
    :param data: исходный массив данных
    :param dt: интервал времени для вычисления ускорений (по умолчанию 1)
    :return: новый массив данных с добавленными ускорениями
    """
    years = 30
    num_bodies = 3

    # Новый массив, который будет на одну строку короче, так как исключаем первый год - не можем вычислить разницу
    original_dofs = 4 # 4 столбца: x, y, vx, vy
    extended_dofs = 4 # 6 столбцов: x, y, vx, vy, ax, ay
    new_data = np.zeros((data.shape[0], num_bodies * extended_dofs * (years - 1)))

    # Заполняем измерения построчно
    for i in range(data.shape[0]):
        row = []
        # для каждого года, но на один меньше
        for year in range(1, years):
            bodies_in_year = []
            # для каждого тедла
            for body in range(num_bodies):
                # Индексы текущего тела
                year_shift = num_bodies * original_dofs * year
                body_shift = body * original_dofs
                idx_x = body_shift + year_shift + 0
                idx_y = body_shift + year_shift + 1
                idx_vx = body_shift + year_shift + 2
                idx_vy = body_shift + year_shift + 3

                # Текущие координаты и скорости
                x, y = data[i, idx_x], data[i, idx_y]
                vx, vy = data[i, idx_vx], data[i, idx_vy]

                # Предыдущие скорости
                prev_vx = data[i, idx_vx - num_bodies * original_dofs]
                prev_vy = data[i, idx_vy - num_bodies * original_dofs]

                # Вычисляем ускорения
                ax = (vx - prev_vx) / dt
                ay = (vy - prev_vy) / dt

                # Добавляем координаты, скорости и ускорения для текущего тела
                # bodies_in_year.extend([x, y, vx, vy, ax, ay])
                bodies_in_year.extend([vx, vy, ax, ay])
            row.extend(bodies_in_year)
        new_data[i] = row  # Заполняем новую строку

    return new_data


def load_data(train_csv_path: str, val_csv_path: str, test_csv_path: str):
    """
    Функция для загрузки данных
    :param train_csv_path: путь к трейновому датасету
    :param val_csv_path: путь к тренировочному датасету
    :param test_csv_path: путь к тестовому датасету
    :return:
    """
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Выделяем фичи и таргет
    feature_columns = [col for col in train_df.columns if col.startswith('y')]
    X_train = train_df[feature_columns].values
    y_train = train_df[['order0']].values.squeeze()

    X_val = val_df[feature_columns].values
    y_val = val_df[['order0']].values.squeeze()

    X_test = test_df[feature_columns].values

    # Добаляем ускорения
    X_train = add_acceleration(X_train)
    X_val = add_acceleration(X_val)
    X_test = add_acceleration(X_test)

    # применияем нормализацию
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Преобразуем данные в тензоры
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, X_test


def init_model(input_size: int, hidden_size: int, output_size: int, lr: float):
    """
    Функция для инициализации модели
    :param lr: learning rate
    :param input_size: размер входного слоя
    :param hidden_size: размер скрытого слоя
    :param output_size: размер выходного слоя
    :return:
    """
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    return model, criterion, optimizer, scheduler


def evaluate(model: MLP, X_test):
    """
    Считаем предикт на заданном датасете
    :param model: модель
    :param X_test: тестовый датасет
    :return:
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predicted_labels = torch.argmax(predictions, dim=1)
    return predicted_labels.numpy()


def train(model, criterion, optimizer, scheduler, X_train, y_train, X_val, y_val, num_epochs, batch_size):
    """
    Обучаем модель
    :param model: модель
    :param criterion: критерий
    :param optimizer: оптимизатор
    :param X_train: трейновый датасет фичей
    :param y_train: трейновый набор меток
    :param X_val: валидационный датасет фичей
    :param y_val: валидационный набор меток
    :param num_epochs: количество эпох
    :param batch_size: размер батча
    :return:
    """
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            optimizer.zero_grad()  # Сброс градиентов
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            scheduler.step(val_loss)

            # Предсказания на валидации
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_accuracy = accuracy_score(y_val.numpy(), val_predictions.numpy())
            # conf_matrix = confusion_matrix(y_val.numpy(), val_predictions.numpy())

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        # print('Confusion Matrix:\n', conf_matrix)


def main(args):
    ### YOUR CODE HERE
    hidden_size = 128
    output_size = 3
    # Load data
    X_train, y_train, X_val, y_val, X_test = load_data(args.train_csv, args.val_csv, args.test_csv)

    # Initialize model
    model, https://discord.gg/3e2AQzvEcriterion, optimizer, scheduler = init_model(X_train.size(1), hidden_size, output_size, args.lr)
    # Train model
    train(model, criterion, optimizer, scheduler, X_train, y_train, X_val, y_val, args.num_epoches, args.batch_size)
    # Predict on test set
    # dump predictions to 'submission.csv'
    submission_df = pd.DataFrame(evaluate(model, X_test))
    submission_df.to_csv(args.out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv', default='../data/train.csv')
    parser.add_argument('--val_csv', default='../data/val.csv')
    parser.add_argument('--test_csv', default='../data/test.csv')
    parser.add_argument('--out_csv', default='../data/submission.csv')
    parser.add_argument('--lr', default=0.01)
    parser.add_argument('--batch_size', default=1024)
    parser.add_argument('--num_epoches', default=30)

    args = parser.parse_args()
    main(args)
