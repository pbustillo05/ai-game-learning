#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <climits>
#include <vector>

using torch::Tensor;

namespace {

struct Tables {
  const int64_t* work0;
  const int64_t* work1;
  const int64_t* trace;
  const int64_t* cost;
  const int64_t* delta;
  const int64_t* head_v;
  const int64_t* head_t;
  const int64_t* head_c;
  const int64_t* sigma_v;
  const int64_t* sigma_t;
  const int64_t* sigma_c;
  const int64_t* action_p;
  const int64_t* action_a;
  const int32_t* work_cap;
  const int32_t* head_cap;
  const int32_t* width;
  const int32_t* input_output;
  const int32_t* conversion;
};

__device__ inline float* pointer(const int64_t* table, int index) {
  return reinterpret_cast<float*>(table[index]);
}

__device__ inline int decode_arena(int encoded) {
  return encoded < 0 ? 1 : 0;
}

__device__ inline int decode_row(int encoded) {
  return encoded < 0 ? -encoded - 1 : encoded;
}

__device__ inline float work_get(
    const int64_t* table, const Tables& t, int output, int lane, int encoded,
    int p, int component, int p_train) {
  if (!table) return 0.0f;
  const int row = decode_row(encoded);
  const int arena = decode_arena(encoded);
  const int cap = t.work_cap[output];
  const int width = t.width[output];
  return pointer(table, output)[
      ((((lane * cap + row) * 2 + arena) * p_train + p) * width) + component];
}

__device__ inline void work_set(
    const int64_t* table, const Tables& t, int output, int lane, int row,
    int arena, int p, int component, int p_train, float value) {
  if (!table) return;
  const int cap = t.work_cap[output];
  const int width = t.width[output];
  pointer(table, output)[
      ((((lane * cap + row) * 2 + arena) * p_train + p) * width) + component] =
      value;
}

__device__ float converted(
    const int64_t* table, const Tables& t, int source, int destination,
    int lane, int encoded, int p, int component, int p_train,
    int output_count, int max_width) {
  if (source == destination) {
    return work_get(table, t, source, lane, encoded, p, component, p_train);
  }
  float total = 0.0f;
  const int source_width = t.width[source];
  const int base = (source * output_count + destination) * max_width;
  for (int h = 0; h < source_width; ++h) {
    if (t.conversion[base + h] == component) {
      total += work_get(table, t, source, lane, encoded, p, h, p_train);
    }
  }
  return total;
}

__device__ inline float scalar_work(
    const int64_t* table, const Tables& t, int output, int lane, int encoded) {
  if (!table) return 0.0f;
  const int row = decode_row(encoded);
  const int arena = decode_arena(encoded);
  return pointer(table, output)[(lane * t.work_cap[output] + row) * 2 + arena];
}

__device__ inline float trace_get(
    const Tables& t, int output, int lane, int encoded, int p, int p_train) {
  const int row = decode_row(encoded);
  const int arena = decode_arena(encoded);
  return pointer(t.trace, output)[
      (((lane * t.work_cap[output] + row) * 2 + arena) * p_train) + p];
}

__device__ inline void trace_set(
    const Tables& t, int output, int lane, int row, int arena, int p,
    int p_train, float value) {
  pointer(t.trace, output)[
      (((lane * t.work_cap[output] + row) * 2 + arena) * p_train) + p] = value;
}

__device__ inline void scalar_set(
    const int64_t* table, const Tables& t, int output, int lane, int row,
    int arena, float value) {
  if (table) {
    pointer(table, output)[(lane * t.work_cap[output] + row) * 2 + arena] = value;
  }
}

__global__ void layered_target_kernel(
    int first, int node_count,
    const int32_t* replica, const int32_t* child_begin,
    const int32_t* child_count, const int32_t* destination_row,
    const int32_t* input_type, const int32_t* output_type,
    const int32_t* owner_slot, const int32_t* nominal_degree,
    const int32_t* target_degree, const int32_t* sigma_head_row,
    const int32_t* sigma_target_row, const int32_t* action_begin,
    const int32_t* child_source_row, const int32_t* child_source_type,
    const int32_t* child_path, const float* child_multiplier,
    Tables tables, int32_t* failure_min, int input_count, int output_count,
    int action_partition_count, int p_train, int max_width, float lambda,
    bool sampled, bool branched, int destination_arena) {
  const int node = first + blockIdx.x;
  if (blockIdx.x >= node_count) return;
  const int lane = replica[node];
  if (failure_min[lane] != INT_MAX) return;
  const int begin = child_begin[node];
  const int count = child_count[node];
  const int ntgt = target_degree[node];
  const int output = output_type[node];
  const int input = input_type[node];
  const int owner = owner_slot[node];
  const int width = tables.width[output];
  bool bad = false;

  for (int linear = threadIdx.x; linear < p_train * width;
       linear += blockDim.x) {
    const int p = linear / width;
    const int d = linear % width;
    float mass_sum = 0.0f;
    float moment_sum = 0.0f;
    for (int a = 0; a < ntgt; ++a) {
      const int edge = begin + a;
      const int source = child_source_type[edge];
      const int encoded = child_source_row[edge];
      mass_sum += converted(
          tables.work0, tables, source, output, lane, encoded, p, d, p_train,
          output_count, max_width);
      if (!sampled) {
        moment_sum += converted(
            tables.work1, tables, source, output, lane, encoded, p, d, p_train,
            output_count, max_width);
      }
    }
    float parent0 = mass_sum / static_cast<float>(ntgt);
    float parent1 = sampled ? 0.0f : moment_sum / static_cast<float>(ntgt);

    if (p == owner && input >= 0) {
      const int head_cap = tables.head_cap[input];
      const int head_row = sigma_head_row[node];
      const float old_v =
          pointer(tables.head_v, input)[(lane * head_cap + head_row) * width + d];
      float sigma_value = sampled
          ? parent0
          : (parent0 > 0.0f ? parent1 / parent0 : 0.0f);
      pointer(tables.sigma_v, lane * input_count + input)[
          sigma_target_row[node] * width + d] = sigma_value;
      const int action_partition = branched ? output : input;
      float* action_a_ptr =
          pointer(tables.action_a, lane * action_partition_count + action_partition);
      float* action_p_ptr = sampled
          ? nullptr
          : pointer(tables.action_p,
                    lane * action_partition_count + action_partition);
      for (int a = 0; a < ntgt; ++a) {
        const int edge = begin + a;
        const int source = child_source_type[edge];
        const int encoded = child_source_row[edge];
        const float child0 = converted(
            tables.work0, tables, source, output, lane, encoded, p, d, p_train,
            output_count, max_width);
        const float child1 = sampled ? 0.0f : converted(
            tables.work1, tables, source, output, lane, encoded, p, d, p_train,
            output_count, max_width);
        float baseline = old_v;
        if (ntgt > 1) {
          float sibling0 = 0.0f;
          float sibling1 = 0.0f;
          for (int b = 0; b < ntgt; ++b) {
            if (b == a) continue;
            const int sibling_edge = begin + b;
            const int sibling_source = child_source_type[sibling_edge];
            const int sibling_encoded = child_source_row[sibling_edge];
            sibling0 += converted(
                tables.work0, tables, sibling_source, output, lane,
                sibling_encoded, p, d, p_train, output_count, max_width);
            if (!sampled) {
              sibling1 += converted(
                  tables.work1, tables, sibling_source, output, lane,
                  sibling_encoded, p, d, p_train, output_count, max_width);
            }
          }
          const float continuation = sampled
              ? sibling0 / static_cast<float>(ntgt - 1)
              : (sibling0 > 0.0f ? sibling1 / sibling0 : 0.0f);
          baseline = lambda * continuation + (1.0f - lambda) * old_v;
        }
        const float q = sampled ? child0 : (child0 > 0.0f ? child1 / child0 : 0.0f);
        const int target_row = action_begin[node] + a;
        action_a_ptr[target_row * width + d] = q - baseline;
        if (action_p_ptr) action_p_ptr[target_row * width + d] = child0;
        if (!isfinite(q) || !isfinite(baseline) || child0 < 0.0f ||
            (!sampled && child0 == 0.0f && fabsf(child1) > 1.0e-7f)) {
          bad = true;
        }
      }
      if (sampled) {
        parent0 = lambda * parent0 + (1.0f - lambda) * old_v;
      } else {
        parent1 = lambda * parent1 + (1.0f - lambda) * parent0 * old_v;
      }
    }
    if (!isfinite(parent0) || (!sampled && !isfinite(parent1)) ||
        (!sampled && parent0 < 0.0f) ||
        (!sampled && parent0 == 0.0f && fabsf(parent1) > 1.0e-7f)) {
      bad = true;
    }
    work_set(
        tables.work0, tables, output, lane, destination_row[node],
        destination_arena, p, d, p_train, parent0);
    if (!sampled) {
      work_set(
          tables.work1, tables, output, lane, destination_row[node],
          destination_arena, p, d, p_train, parent1);
    }
  }

  if (branched && threadIdx.x == 0) {
    float delta_bar = 0.0f;
    float cost_raw = 0.0f;
    for (int a = 0; a < ntgt; ++a) {
      const int edge = begin + a;
      const int source = child_source_type[edge];
      const int encoded = child_source_row[edge];
      delta_bar += scalar_work(tables.delta, tables, source, lane, encoded);
      cost_raw += static_cast<float>(child_path[edge]) +
          child_multiplier[edge] *
              scalar_work(tables.cost, tables, source, lane, encoded);
    }
    delta_bar /= static_cast<float>(ntgt);
    cost_raw /= static_cast<float>(ntgt);
    float trace_exec = 0.0f;
    for (int p = 0; p < p_train; ++p) {
      float mean = 0.0f;
      for (int a = 0; a < ntgt; ++a) {
        const int edge = begin + a;
        mean += trace_get(
            tables, child_source_type[edge], lane, child_source_row[edge], p,
            p_train);
      }
      mean /= static_cast<float>(ntgt);
      trace_set(
          tables, output, lane, destination_row[node], destination_arena, p,
          p_train, mean);
      if (ntgt > 1) {
        for (int a = 0; a < ntgt; ++a) {
          const int edge = begin + a;
          const float value = trace_get(
              tables, child_source_type[edge], lane, child_source_row[edge], p,
              p_train);
          const float difference = value - mean;
          trace_exec += difference * difference;
        }
      }
    }
    if (ntgt > 1) trace_exec /= static_cast<float>(ntgt - 1);
    float propagated_delta = delta_bar / static_cast<float>(ntgt);
    float propagated_cost = cost_raw;
    if (input >= 0) {
      const int head_index =
          lane * tables.head_cap[input] + sigma_head_row[node];
      const float old_t = pointer(tables.head_t, input)[head_index];
      const float old_c = pointer(tables.head_c, input)[head_index];
      const float trace_hat = ntgt > 1 ? fmaxf(0.0f, trace_exec - delta_bar) : 0.0f;
      const float t_star = ntgt > 1 ? trace_hat : old_t;
      propagated_delta =
          (t_star + delta_bar) / static_cast<float>(ntgt) -
          t_star / static_cast<float>(nominal_degree[node]);
      propagated_cost = lambda * cost_raw + (1.0f - lambda) * old_c;
      pointer(tables.sigma_t, lane * input_count + input)[sigma_target_row[node]] =
          trace_hat;
      pointer(tables.sigma_c, lane * input_count + input)[sigma_target_row[node]] =
          cost_raw;
    }
    scalar_set(
        tables.cost, tables, output, lane, destination_row[node],
        destination_arena, propagated_cost);
    scalar_set(
        tables.delta, tables, output, lane, destination_row[node],
        destination_arena, propagated_delta);
    if (!isfinite(trace_exec) || !isfinite(propagated_cost) ||
        !isfinite(propagated_delta)) bad = true;
  }
  if (bad) atomicMin(failure_min + lane, node);
}

Tensor pointer_tensor(const std::vector<Tensor>& tensors, const Tensor& like) {
  std::vector<int64_t> values;
  values.reserve(tensors.size());
  for (const Tensor& tensor : tensors) {
    values.push_back(tensor.numel() ? reinterpret_cast<int64_t>(tensor.data_ptr()) : 0);
  }
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt64))
      .to(like.device());
}

Tensor int_tensor(const std::vector<int64_t>& values, const Tensor& like) {
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kInt32))
      .to(like.device());
}

}  // namespace

Tensor run_layers(
    Tensor layer_ptr_cpu, std::vector<Tensor> nodes, std::vector<Tensor> children,
    std::vector<Tensor> work0, std::vector<Tensor> work1,
    std::vector<Tensor> trace, std::vector<Tensor> cost,
    std::vector<Tensor> delta, std::vector<Tensor> head_v,
    std::vector<Tensor> head_t, std::vector<Tensor> head_c,
    std::vector<Tensor> sigma_v, std::vector<Tensor> sigma_t,
    std::vector<Tensor> sigma_c, std::vector<Tensor> action_p,
    std::vector<Tensor> action_a, Tensor conversion,
    std::vector<int64_t> work_cap, std::vector<int64_t> head_cap,
    std::vector<int64_t> width, std::vector<int64_t> input_output,
    int64_t n_replica, int64_t input_count, int64_t output_count,
    int64_t action_partition_count, int64_t p_train, int64_t max_width,
    double lambda, bool sampled, bool branched) {
  TORCH_CHECK(layer_ptr_cpu.device().is_cpu(), "layer_ptr must be CPU");
  const Tensor& like = nodes[0];
  c10::cuda::CUDAGuard guard(like.device());
  Tensor failure = torch::full(
      {n_replica}, INT_MAX,
      torch::TensorOptions().dtype(torch::kInt32).device(like.device()));
  Tensor work0_ptr = pointer_tensor(work0, like);
  Tensor work1_ptr = pointer_tensor(work1, like);
  Tensor trace_ptr = pointer_tensor(trace, like);
  Tensor cost_ptr = pointer_tensor(cost, like);
  Tensor delta_ptr = pointer_tensor(delta, like);
  Tensor head_v_ptr = pointer_tensor(head_v, like);
  Tensor head_t_ptr = pointer_tensor(head_t, like);
  Tensor head_c_ptr = pointer_tensor(head_c, like);
  Tensor sigma_v_ptr = pointer_tensor(sigma_v, like);
  Tensor sigma_t_ptr = pointer_tensor(sigma_t, like);
  Tensor sigma_c_ptr = pointer_tensor(sigma_c, like);
  Tensor action_p_ptr = pointer_tensor(action_p, like);
  Tensor action_a_ptr = pointer_tensor(action_a, like);
  Tensor work_cap_t = int_tensor(work_cap, like);
  Tensor head_cap_t = int_tensor(head_cap, like);
  Tensor width_t = int_tensor(width, like);
  Tensor input_output_t = int_tensor(input_output, like);
  Tables tables{
      work0_ptr.data_ptr<int64_t>(), work1_ptr.data_ptr<int64_t>(),
      trace_ptr.data_ptr<int64_t>(), cost_ptr.data_ptr<int64_t>(),
      delta_ptr.data_ptr<int64_t>(), head_v_ptr.data_ptr<int64_t>(),
      head_t_ptr.data_ptr<int64_t>(), head_c_ptr.data_ptr<int64_t>(),
      sigma_v_ptr.data_ptr<int64_t>(), sigma_t_ptr.data_ptr<int64_t>(),
      sigma_c_ptr.data_ptr<int64_t>(), action_p_ptr.data_ptr<int64_t>(),
      action_a_ptr.data_ptr<int64_t>(), work_cap_t.data_ptr<int32_t>(),
      head_cap_t.data_ptr<int32_t>(), width_t.data_ptr<int32_t>(),
      input_output_t.data_ptr<int32_t>(), conversion.data_ptr<int32_t>()};
  const int32_t* ptr = layer_ptr_cpu.data_ptr<int32_t>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  for (int layer = 0; layer + 1 < layer_ptr_cpu.numel(); ++layer) {
    const int first = ptr[layer];
    const int count = ptr[layer + 1] - first;
    if (!count) continue;
    layered_target_kernel<<<count, 128, 0, stream>>>(
        first, count,
        nodes[0].data_ptr<int32_t>(), nodes[1].data_ptr<int32_t>(),
        nodes[2].data_ptr<int32_t>(), nodes[3].data_ptr<int32_t>(),
        nodes[4].data_ptr<int32_t>(), nodes[5].data_ptr<int32_t>(),
        nodes[6].data_ptr<int32_t>(), nodes[7].data_ptr<int32_t>(),
        nodes[8].data_ptr<int32_t>(), nodes[9].data_ptr<int32_t>(),
        nodes[10].data_ptr<int32_t>(), nodes[11].data_ptr<int32_t>(),
        children[0].data_ptr<int32_t>(), children[1].data_ptr<int32_t>(),
        children[2].data_ptr<int32_t>(), children[3].data_ptr<float>(),
        tables, failure.data_ptr<int32_t>(), input_count, output_count,
        action_partition_count, p_train, max_width, lambda, sampled, branched,
        (layer + 1) & 1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return failure;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_layers", &run_layers, "Queued layered target CUDA executor");
}
