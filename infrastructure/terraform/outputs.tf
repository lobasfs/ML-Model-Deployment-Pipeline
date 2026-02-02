output "cluster_endpoint" {
  description = "Kubernetes cluster endpoint"
  value       = module.kubernetes.cluster_endpoint
}

output "models_bucket" {
  description = "S3 bucket for models"
  value       = module.storage.models_bucket
}

output "data_bucket" {
  description = "S3 bucket for data (DVC)"
  value       = module.storage.data_bucket
}

output "kubeconfig_command" {
  description = "Command to get kubeconfig"
  value       = "yandex managed-kubernetes cluster get-credentials ${module.kubernetes.cluster_id} --external"
}
