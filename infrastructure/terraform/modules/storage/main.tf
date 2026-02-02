# Storage Module - Object Storage для моделей и данных

# Service Account для Object Storage
resource "yandex_iam_service_account" "storage_sa" {
  name        = "${var.project_name}-storage-sa"
  description = "Service account for Object Storage"
}

# Статический ключ доступа
resource "yandex_iam_service_account_static_access_key" "storage_key" {
  service_account_id = yandex_iam_service_account.storage_sa.id
  description        = "Static access key for Object Storage"
}

# Роли для Storage SA
resource "yandex_resourcemanager_folder_iam_member" "storage_editor" {
  folder_id = var.folder_id
  role      = "storage.editor"
  member    = "serviceAccount:${yandex_iam_service_account.storage_sa.id}"
}

# Bucket для моделей
resource "yandex_storage_bucket" "models" {
  bucket     = "${var.project_name}-models"
  access_key = yandex_iam_service_account_static_access_key.storage_key.access_key
  secret_key = yandex_iam_service_account_static_access_key.storage_key.secret_key

  versioning {
    enabled = true
  }

  lifecycle_rule {
    id      = "delete-old-versions"
    enabled = true

    noncurrent_version_expiration {
      days = 90
    }
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "aws:kms"
      }
    }
  }
}

# Bucket для данных (DVC)
resource "yandex_storage_bucket" "data" {
  bucket     = "${var.project_name}-data"
  access_key = yandex_iam_service_account_static_access_key.storage_key.access_key
  secret_key = yandex_iam_service_account_static_access_key.storage_key.secret_key

  versioning {
    enabled = true
  }

  lifecycle_rule {
    id      = "archive-old-data"
    enabled = true

    transition {
      days          = 30
      storage_class = "COLD"
    }
  }
}

# Bucket для логов
resource "yandex_storage_bucket" "logs" {
  bucket     = "${var.project_name}-logs"
  access_key = yandex_iam_service_account_static_access_key.storage_key.access_key
  secret_key = yandex_iam_service_account_static_access_key.storage_key.secret_key

  lifecycle_rule {
    id      = "expire-logs"
    enabled = true

    expiration {
      days = 30
    }
  }
}

output "models_bucket" {
  value = yandex_storage_bucket.models.bucket
}

output "data_bucket" {
  value = yandex_storage_bucket.data.bucket
}

output "access_key" {
  value     = yandex_iam_service_account_static_access_key.storage_key.access_key
  sensitive = true
}

output "secret_key" {
  value     = yandex_iam_service_account_static_access_key.storage_key.secret_key
  sensitive = true
}
