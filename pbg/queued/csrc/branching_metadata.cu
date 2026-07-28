#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

using torch::Tensor;

__global__ void decision_log_mass_kernel(
    const int32_t* parent,
    const int32_t* degree,
    float* log_mass,
    int32_t* failure,
    int64_t size) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (int64_t node = 0; node < size; ++node) {
    const int32_t ancestor = parent[node];
    if (ancestor >= node) {
      failure[0] = 1;
      return;
    }
    if (ancestor < 0) {
      log_mass[node] = 0.0f;
      continue;
    }
    const int32_t ancestor_degree = degree[ancestor];
    if (ancestor_degree <= 0) {
      failure[0] = 2;
      return;
    }
    log_mass[node] = log_mass[ancestor] - logf(static_cast<float>(ancestor_degree));
  }
}

Tensor decision_log_mass(Tensor parent, Tensor degree, Tensor failure) {
  TORCH_CHECK(parent.is_cuda(), "parent must be CUDA");
  TORCH_CHECK(parent.scalar_type() == torch::kInt32, "parent must be int32");
  TORCH_CHECK(parent.dim() == 1 && parent.is_contiguous(), "parent must be contiguous [N]");
  TORCH_CHECK(
      degree.is_cuda() && degree.scalar_type() == torch::kInt32 &&
          degree.sizes() == parent.sizes() && degree.is_contiguous(),
      "degree must be contiguous CUDA int32 [N]");
  TORCH_CHECK(
      failure.is_cuda() && failure.scalar_type() == torch::kInt32 &&
          failure.numel() == 1 && failure.is_contiguous(),
      "failure must be one contiguous CUDA int32");
  TORCH_CHECK(
      degree.device() == parent.device() && failure.device() == parent.device(),
      "branching metadata tensors must share one CUDA device");
  c10::cuda::CUDAGuard guard(parent.device());
  Tensor result = torch::empty(
      parent.sizes(),
      torch::TensorOptions().dtype(torch::kFloat32).device(parent.device()));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  decision_log_mass_kernel<<<1, 1, 0, stream>>>(
      parent.data_ptr<int32_t>(),
      degree.data_ptr<int32_t>(),
      result.data_ptr<float>(),
      failure.data_ptr<int32_t>(),
      parent.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decision_log_mass", &decision_log_mass, "Propagate decision log mass on CUDA");
}
