# Многостадийная сборка для обучения модели

# Стадия 1: Базовый образ с зависимостями
FROM python:3.10-slim as base

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Стадия 2: Обучение
FROM base as training

# Копирование исходного кода
COPY src/ ./src/
COPY main.py .
COPY config.yaml .

# DVC для управления данными
RUN pip install dvc[s3]
COPY .dvc/ ./.dvc/

# Настройка DVC remote (example for S3-compatible storage)
RUN dvc remote add -d storage s3://ml-models-bucket/data

# Команда для обучения
CMD ["python", "main.py"]
