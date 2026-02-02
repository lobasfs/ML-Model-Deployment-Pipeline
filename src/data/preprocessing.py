import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class DataPreprocessor:
    """Класс для предобработки данных"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = None

    def load_data(self, data_path):
        """Загрузка данных из CSV"""
        df = pd.read_csv(data_path)
        return df

    def prepare_features(self, df, target_col='default.payment.next.month'):
        """Подготовка признаков и целевой переменной"""
        self.feature_cols = [col for col in df.columns
                             if col not in ['ID', target_col]]
        X = df[self.feature_cols].values
        y = df[target_col].values
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Разделение на train/test"""
        return train_test_split(
            X, y, test_size=test_size,
            random_state=random_state, stratify=y
        )

    def scale_data(self, X_train, X_test):
        """Нормализация данных"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def reshape_for_lstm(self, X):
        """Преобразование в формат для LSTM (batch, seq_len, features)"""
        return X.reshape(-1, 1, X.shape[1])

    def full_pipeline(self, data_path):
        """Полный пайплайн обработки данных"""
        df = self.load_data(data_path)
        X, y = self.prepare_features(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train_scaled, X_test_scaled = self.scale_data(X_train, X_test)
        X_train_lstm = self.reshape_for_lstm(X_train_scaled)
        X_test_lstm = self.reshape_for_lstm(X_test_scaled)

        return {
            'X_train': X_train_lstm,
            'X_test': X_test_lstm,
            'y_train': y_train,
            'y_test': y_test,
            'input_size': len(self.feature_cols)
        }

    def save_scaler(self, path='scaler.pkl'):
        """Сохранение scaler для production"""
        joblib.dump(self.scaler, path)

    def load_scaler(self, path='scaler.pkl'):
        """Загрузка scaler"""
        self.scaler = joblib.load(path)
