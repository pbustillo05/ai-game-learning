#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <climits>

using torch::Tensor;

__global__ void alias_table_kernel(
    double* scaled,
    double* probability,
    int32_t* alias,
    int64_t* small,
    int64_t* large,
    int64_t size) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int64_t small_count = 0;
  int64_t large_count = 0;
  for (int64_t index = 0; index < size; ++index) {
    probability[index] = 1.0;
    alias[index] = static_cast<int32_t>(index);
    if (scaled[index] < 1.0) {
      small[small_count++] = index;
    } else {
      large[large_count++] = index;
    }
  }
  while (small_count != 0 && large_count != 0) {
    const int64_t low = small[--small_count];
    const int64_t high = large[--large_count];
    const double low_probability = scaled[low];
    probability[low] = low_probability;
    alias[low] = static_cast<int32_t>(high);
    scaled[high] -= 1.0 - low_probability;
    if (scaled[high] < 1.0) {
      small[small_count++] = high;
    } else {
      large[large_count++] = high;
    }
  }
}

void build_alias_table(
    Tensor scaled,
    Tensor probability,
    Tensor alias,
    Tensor small,
    Tensor large) {
  TORCH_CHECK(scaled.is_cuda(), "scaled weights must be CUDA");
  TORCH_CHECK(scaled.scalar_type() == torch::kFloat64, "scaled weights must be FP64");
  TORCH_CHECK(scaled.dim() == 1 && scaled.numel() > 0, "scaled weights must be [N]");
  TORCH_CHECK(scaled.is_contiguous(), "scaled weights must be contiguous");
  const int64_t size = scaled.numel();
  TORCH_CHECK(
      probability.is_cuda() && probability.scalar_type() == torch::kFloat64 &&
          probability.sizes() == scaled.sizes() && probability.is_contiguous(),
      "probability must be contiguous CUDA FP64 [N]");
  TORCH_CHECK(
      alias.is_cuda() && alias.scalar_type() == torch::kInt32 &&
          alias.sizes() == scaled.sizes() && alias.is_contiguous(),
      "alias must be contiguous CUDA int32 [N]");
  TORCH_CHECK(
      small.is_cuda() && small.scalar_type() == torch::kInt64 &&
          small.sizes() == scaled.sizes() && small.is_contiguous(),
      "small workspace must be contiguous CUDA int64 [N]");
  TORCH_CHECK(
      large.is_cuda() && large.scalar_type() == torch::kInt64 &&
          large.sizes() == scaled.sizes() && large.is_contiguous(),
      "large workspace must be contiguous CUDA int64 [N]");
  TORCH_CHECK(
      probability.device() == scaled.device() && alias.device() == scaled.device() &&
          small.device() == scaled.device() && large.device() == scaled.device(),
      "alias-table tensors must share one CUDA device");
  TORCH_CHECK(
      size <= static_cast<int64_t>(INT32_MAX),
      "alias-table size exceeds int32 indexing");
  c10::cuda::CUDAGuard guard(scaled.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  alias_table_kernel<<<1, 1, 0, stream>>>(
      scaled.data_ptr<double>(),
      probability.data_ptr<double>(),
      alias.data_ptr<int32_t>(),
      small.data_ptr<int64_t>(),
      large.data_ptr<int64_t>(),
      size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_alias_table", &build_alias_table, "Build a Walker alias table on CUDA");
}
