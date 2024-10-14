import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

print("GPU: ", torch.cuda.is_available())


class MLP(nn.Module):
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


def load_data(train_csv : str, val_csv : str, test_csv : str):
    """
    :param train_csv: трейновый датасет
    :param val_csv: валидационный датасет
    :param test_csv: тестовый датасет
    :return:
    """
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # Выделяем фичи и таргет
    feature_columns = [col for col in train_df.columns if col.startswith('y')]
    X_train = train_df[feature_columns].values
    y_train = train_df[['order0']].values.squeeze()

    X_val = val_df[feature_columns].values
    y_val = val_df[['order0']].values.squeeze()

    X_test = test_df[feature_columns].values

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


def init_model(input_size: int, hidden_size: int, output_size: int):
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, optimizer


def evaluate(model, X_test):
    ### YOUR CODE HERE
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predicted_labels = torch.argmax(predictions, dim=1)
    return predicted_labels.numpy()

def train(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs, batch_size):
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

            # Предсказания на валидации
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_accuracy = accuracy_score(y_val.numpy(), val_predictions.numpy())
            conf_matrix = confusion_matrix(y_val.numpy(), val_predictions.numpy())

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        print('Confusion Matrix:\n', conf_matrix)


def main(args):
    ### YOUR CODE HERE
    hidden_size = 128
    output_size = 3
    num_epochs = 20
    batch_size = 1024
    # Load data
    X_train, y_train, X_val, y_val, X_test = load_data('../data/train.csv', '../data/val.csv', '../data/test.csv')
    # Initialize model
    model, criterion, optimizer = init_model(X_train.size(1), hidden_size, output_size)
    # Train model
    train(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs, batch_size)
    # Predict on test set
    # dump predictions to 'submission.csv'
    submission_df = pd.DataFrame(evaluate(model, X_test))
    submission_df.to_csv('../data/submission.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv', default='homeworks/hw1/data/train.csv')
    parser.add_argument('--val_csv', default='homeworks/hw1/data/val.csv')
    parser.add_argument('--test_csv', default='homeworks/hw1/data/test.csv')
    parser.add_argument('--out_csv', default='homeworks/hw1/data/submission.csv')
    parser.add_argument('--lr', default=0)
    parser.add_argument('--batch_size', default=0)
    parser.add_argument('--num_epoches', default=0)

    args = parser.parse_args()
    main(args)