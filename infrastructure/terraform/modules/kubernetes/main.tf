# Kubernetes Module

# Service Account для Kubernetes
resource "yandex_iam_service_account" "k8s_sa" {
  name        = "${var.project_name}-k8s-sa"
  description = "Service account for Kubernetes cluster"
}

# Роли для Service Account
resource "yandex_resourcemanager_folder_iam_member" "k8s_editor" {
  folder_id = var.folder_id
  role      = "editor"
  member    = "serviceAccount:${yandex_iam_service_account.k8s_sa.id}"
}

resource "yandex_resourcemanager_folder_iam_member" "k8s_images_puller" {
  folder_id = var.folder_id
  role      = "container-registry.images.puller"
  member    = "serviceAccount:${yandex_iam_service_account.k8s_sa.id}"
}

# Kubernetes кластер
resource "yandex_kubernetes_cluster" "main" {
  name        = "${var.project_name}-cluster"
  description = "Managed Kubernetes cluster"

  network_id = var.network_id

  master {
    version = var.k8s_version

    zonal {
      zone      = var.zone
      subnet_id = var.k8s_subnet_id
    }

    public_ip = true

    security_group_ids = [var.security_group_id]

    maintenance_policy {
      auto_upgrade = true

      maintenance_window {
        day        = "monday"
        start_time = "03:00"
        duration   = "3h"
      }
    }
  }

  service_account_id      = yandex_iam_service_account.k8s_sa.id
  node_service_account_id = yandex_iam_service_account.k8s_sa.id

  release_channel = "STABLE"

  labels = {
    environment = var.environment
    project     = var.project_name
  }

  depends_on = [
    yandex_resourcemanager_folder_iam_member.k8s_editor,
    yandex_resourcemanager_folder_iam_member.k8s_images_puller
  ]
}

# Node Group для CPU workloads
resource "yandex_kubernetes_node_group" "cpu_nodes" {
  cluster_id  = yandex_kubernetes_cluster.main.id
  name        = "${var.project_name}-cpu-nodes"
  description = "CPU node group for general workloads"
  version     = var.k8s_version

  instance_template {
    platform_id = "standard-v3"

    resources {
      cores  = var.node_cpu_count
      memory = var.node_memory_gb
    }

    boot_disk {
      type = "network-ssd"
      size = 64
    }

    scheduling_policy {
      preemptible = false
    }

    network_interface {
      nat                = true
      subnet_ids         = [var.k8s_subnet_id]
      security_group_ids = [var.security_group_id]
    }

    metadata = {
      ssh-keys = "ubuntu:${file("~/.ssh/id_rsa.pub")}"
    }
  }

  scale_policy {
    auto_scale {
      min     = var.node_count
      max     = var.max_node_count
      initial = var.node_count
    }
  }

  allocation_policy {
    location {
      zone = var.zone
    }
  }

  labels = {
    environment = var.environment
    node_type   = "cpu"
  }
}

# # Node Group для GPU workloads (опционально)
# resource "yandex_kubernetes_node_group" "gpu_nodes" {
#   count = var.enable_gpu ? 1 : 0
#
#   cluster_id  = yandex_kubernetes_cluster.main.id
#   name        = "${var.project_name}-gpu-nodes"
#   description = "GPU node group for training workloads"
#   version     = var.k8s_version
#
#   instance_template {
#     platform_id = "gpu-standard-v3"
#
#     resources {
#       cores  = 8
#       memory = 32
#       gpus   = 1
#     }
#
#     boot_disk {
#       type = "network-ssd"
#       size = 128
#     }
#
#     scheduling_policy {
#       preemptible = false
#     }
#
#     network_interface {
#       nat                = true
#       subnet_ids         = [var.k8s_subnet_id]
#       security_group_ids = [var.security_group_id]
#     }
#   }
#
#   scale_policy {
#     fixed_scale {
#       size = 1
#     }
#   }
#
#   allocation_policy {
#     location {
#       zone = var.zone
#     }
#   }
#
#   labels = {
#     environment = var.environment
#     node_type   = "gpu"
#   }
# }
#
# output "cluster_id" {
#   value = yandex_kubernetes_cluster.main.id
# }
#
# output "cluster_endpoint" {
#   value = yandex_kubernetes_cluster.main.master[0].external_v4_endpoint
# }
