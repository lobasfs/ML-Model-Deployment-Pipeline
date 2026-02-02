# ML-Model-Deployment-Pipeline
Комплексный проект развертывания ML-модели с полным CI/CD пайплайном, мониторингом и автоматическим переобучением.

Описание
Проект демонстрирует полный цикл промышленного развертывания модели машинного обучения:

Обучение LSTM модели для предсказания дефолта по кредитным картам
Конвертация в ONNX и оптимизация (квантизация)
Контейнеризация с Docker
Оркестрация в Kubernetes
Infrastructure as Code с Terraform
CI/CD пайплайн с GitHub Actions
Мониторинг с Prometheus и Grafana
Детектирование дрифта с Evidently AI
Автоматическое переобучение с Apache Airflow
Структура проекта
project/
├── api/                     # FastAPI приложение
├── infrastructure/          # Инфраструктура
│   ├── docker/              # докерфайлы
│   ├── kubernetes/          # K8s манифесты
│   └── terraform/           # Terraform конфигурации
├── monitoring/              # Мониторинг
│   ├── grafana/             # Grafana dashboards
│   └── prometheus/          # Prometheus config
├── src/                      # Исходный код
│   ├── benchmarking/        # Бенчмаркинг и нагрузка
│   ├── data/                # Обработка данных
│   ├── deployment/          # Развертывание
│   ├── models/              # Модели
│   ├── training/            # Обучение
│   └── utils/               # Утилиты
├── scripts/                 # Скрипты
│   ├── airflow_dags/
│   └── check_drift.py
├── .github/                 # GitHub Actions
│   └── workflows/
├── models/                  # Сохраненные модели
├── data/                    # Данные
├── reports/                 # Отчеты
├── main.py                  # Главный скрипт
├── config.yaml              # Конфигурация
├── requirements.txt         # Зависимости
└── README.md               # Документация
Infrastructure as Code (Terraform)
Инициализация Terraform
cd infrastructure/terraform

# Инициализация
terraform init

# План
terraform plan -var-file="production.tfvars"

# Применение
terraform apply -var-file="production.tfvars"
Получение kubeconfig
yandex managed-kubernetes cluster get-credentials <cluster-id> --external
CI/CD Pipeline
GitHub Actions автоматически:

Запускает тесты и линтинг
Проверяет безопасность (Trivy, Bandit)
Собирает Docker образы
Развертывает в staging
После одобрения - в production
Для настройки необходимо добавить secrets в GitHub:

KUBE_CONFIG_STAGING
KUBE_CONFIG_PROD
SLACK_WEBHOOK
Мониторинг
Prometheus метрики
model_inference_requests_total - количество запросов
model_inference_duration_seconds - время инференса
model_inference_errors_total - количество ошибок
model_accuracy - текущая точность модели
data_drift_score - score дрифта данных
Автоматическое переобучение
Airflow DAG запускается каждое воскресенье в 2:00:

Проверка качества данных
Детектирование дрифта
Переобучение модели (если дрифт обнаружен)
Оценка новой модели
Сравнение с production
Развертывание (если лучше)
