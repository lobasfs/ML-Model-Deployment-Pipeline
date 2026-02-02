# Monitoring Module

resource "yandex_monitoring_dashboard" "main" {
  name        = "${var.project_name}-dashboard"
  description = "Main monitoring dashboard"

  parametrization {
    parameters {
      id    = "cluster"
      title = "Cluster"
      values = [yandex_kubernetes_cluster.main.id]
    }
  }

  labels = {
    environment = var.environment
  }
}
