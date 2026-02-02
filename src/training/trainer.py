"""
Модуль обучения модели
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path


class ModelTrainer:
    """Класс для обучения модели"""

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.history = {'loss': [], 'val_loss': []}

    def prepare_dataloaders(self, X_train, y_train, X_val=None, y_val=None,
                            batch_size=256):
        """Подготовка DataLoader"""
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train).reshape(-1, 1)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader

    def train_epoch(self, train_loader, criterion, optimizer):
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader, criterion):
        """Валидация модели"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=20, batch_size=256, lr=0.001, verbose=True):
        """Полный цикл обучения"""

        train_loader, val_loader = self.prepare_dataloaders(
            X_train, y_train, X_val, y_val, batch_size
        )

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            self.history['loss'].append(train_loss)

            if val_loader is not None:
                val_loss = self.validate(val_loader, criterion)
                self.history['val_loss'].append(val_loss)

            if verbose and (epoch + 1) % 5 == 0:
                if val_loader is not None:
                    print(f'Epoch [{epoch+1}/{epochs}], '
                          f'Train Loss: {train_loss:.4f}, '
                          f'Val Loss: {val_loss:.4f}')
                else:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}')

        return self.model

    def save_model(self, path='model.pth'):
        """Сохранение модели"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        print(f"✓ Модель сохранена: {path}")

    def load_model(self, path='model.pth'):
        """Загрузка модели"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {'loss': [], 'val_loss': []})
        print(f"✓ Модель загружена: {path}")
