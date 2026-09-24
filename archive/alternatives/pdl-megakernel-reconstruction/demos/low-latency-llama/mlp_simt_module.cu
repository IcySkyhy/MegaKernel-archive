#include "mlp_simt.cu"
#include "qkv_simt.cu"
#include "oproj_simt.cu"
#include "attention_simt.cu"
#include "attention_partial.cu"
#include "attention_tk_short.cu"
#include "attention_reduction_simt.cu"
#include "lm_head_simt.cu"

#include "pyutils/pyutils.cuh"

#include <stdexcept>
#include <vector>

using llama_globals = ::llama_1b_globals;

// CUDA 13 promoted the edge-data graph APIs to the unsuffixed names. CUDA
// 12.x exposes the same signatures as *_v2. Keep the pybind graph helpers on
// one source path without changing their runtime semantics.
static cudaError_t graph_get_edges(cudaGraph_t graph, cudaGraphNode_t *from,
                                   cudaGraphNode_t *to,
                                   cudaGraphEdgeData *edge_data,
                                   size_t *num_edges) {
#if CUDART_VERSION >= 13000
    return cudaGraphGetEdges(graph, from, to, edge_data, num_edges);
#else
    return cudaGraphGetEdges_v2(graph, from, to, edge_data, num_edges);
#endif
}

static cudaError_t graph_add_dependencies(
    cudaGraph_t graph, const cudaGraphNode_t *from,
    const cudaGraphNode_t *to, const cudaGraphEdgeData *edge_data,
    size_t num_dependencies) {
#if CUDART_VERSION >= 13000
    return cudaGraphAddDependencies(
        graph, from, to, edge_data, num_dependencies);
#else
    return cudaGraphAddDependencies_v2(
        graph, from, to, edge_data, num_dependencies);
#endif
}

static cudaError_t graph_remove_dependencies(
    cudaGraph_t graph, const cudaGraphNode_t *from,
    const cudaGraphNode_t *to, const cudaGraphEdgeData *edge_data,
    size_t num_dependencies) {
#if CUDART_VERSION >= 13000
    return cudaGraphRemoveDependencies(
        graph, from, to, edge_data, num_dependencies);
#else
    return cudaGraphRemoveDependencies_v2(
        graph, from, to, edge_data, num_dependencies);
#endif
}

template <auto kernel, int grid_blocks, int block_threads, typename TGlobal>
static void bind_fixed_kernel(auto m, auto name,
                              auto TGlobal::*... member_ptrs) {
    m.def(name,
          [](kittens::py::object<decltype(member_ptrs)>... args,
             pybind11::kwargs kwargs) {
              TGlobal g{
                  kittens::py::from_object<typename kittens::py::trait<
                      decltype(member_ptrs)>::member_type>::make(args)...};
              cudaStream_t raw_stream = nullptr;
              if (kwargs.contains("stream")) {
                  const uintptr_t stream_ptr =
                      kwargs["stream"].attr("cuda_stream").cast<uintptr_t>();
                  raw_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
              }
              kernel<<<grid_blocks, block_threads, 0, raw_stream>>>(g);
          });
}

template <auto kernel, int grid_blocks, int block_threads, typename TGlobal>
static void bind_fixed_kernel_pdl(auto m, auto name,
                                  auto TGlobal::*... member_ptrs) {
    m.def(name,
          [](kittens::py::object<decltype(member_ptrs)>... args,
             pybind11::kwargs kwargs) {
              TGlobal g{
                  kittens::py::from_object<typename kittens::py::trait<
                      decltype(member_ptrs)>::member_type>::make(args)...};
              cudaStream_t raw_stream = nullptr;
              if (kwargs.contains("stream")) {
                  const uintptr_t stream_ptr =
                      kwargs["stream"].attr("cuda_stream").cast<uintptr_t>();
                  raw_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
              }
              cudaLaunchConfig_t config{};
              config.gridDim = dim3(grid_blocks);
              config.blockDim = dim3(block_threads);
              config.dynamicSmemBytes = 0;
              config.stream = raw_stream;
              cudaLaunchAttribute attribute{};
              attribute.id =
                  cudaLaunchAttributeProgrammaticStreamSerialization;
              attribute.val.programmaticStreamSerializationAllowed = 1;
              config.attrs = &attribute;
              config.numAttrs = 1;
              const cudaError_t status =
                  cudaLaunchKernelEx(&config, kernel, g);
              if (status != cudaSuccess) {
                  throw std::runtime_error(cudaGetErrorString(status));
              }
          });
}

template <auto kernel, int grid_blocks, int block_threads,
          int dynamic_smem_bytes, typename TGlobal>
static void bind_fixed_kernel_smem(auto m, auto name,
                                   auto TGlobal::*... member_ptrs) {
    const cudaError_t attr_status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_smem_bytes);
    if (attr_status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(attr_status));
    }
    m.def(name,
          [](kittens::py::object<decltype(member_ptrs)>... args,
             pybind11::kwargs kwargs) {
              TGlobal g{
                  kittens::py::from_object<typename kittens::py::trait<
                      decltype(member_ptrs)>::member_type>::make(args)...};
              cudaStream_t raw_stream = nullptr;
              if (kwargs.contains("stream")) {
                  const uintptr_t stream_ptr =
                      kwargs["stream"].attr("cuda_stream").cast<uintptr_t>();
                  raw_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
              }
              kernel<<<grid_blocks, block_threads, dynamic_smem_bytes,
                       raw_stream>>>(g);
          });
}

template <auto kernel, int grid_blocks, int block_threads,
          int dynamic_smem_bytes, typename TGlobal>
static void bind_fixed_kernel_smem_pdl(auto m, auto name,
                                       auto TGlobal::*... member_ptrs) {
    const cudaError_t attr_status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_smem_bytes);
    if (attr_status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(attr_status));
    }
    m.def(name,
          [](kittens::py::object<decltype(member_ptrs)>... args,
             pybind11::kwargs kwargs) {
              TGlobal g{
                  kittens::py::from_object<typename kittens::py::trait<
                      decltype(member_ptrs)>::member_type>::make(args)...};
              cudaStream_t raw_stream = nullptr;
              if (kwargs.contains("stream")) {
                  const uintptr_t stream_ptr =
                      kwargs["stream"].attr("cuda_stream").cast<uintptr_t>();
                  raw_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
              }
              cudaLaunchConfig_t config{};
              config.gridDim = dim3(grid_blocks);
              config.blockDim = dim3(block_threads);
              config.dynamicSmemBytes = dynamic_smem_bytes;
              config.stream = raw_stream;
              cudaLaunchAttribute attribute{};
              attribute.id =
                  cudaLaunchAttributeProgrammaticStreamSerialization;
              attribute.val.programmaticStreamSerializationAllowed = 1;
              config.attrs = &attribute;
              config.numAttrs = 1;
              const cudaError_t status =
                  cudaLaunchKernelEx(&config, kernel, g);
              if (status != cudaSuccess) {
                  throw std::runtime_error(cudaGetErrorString(status));
              }
          });
}

#define LLAMA_GLOBAL_MEMBERS                                                  \
    &llama_globals::Bar, &llama_globals::instructions,                        \
        &llama_globals::timings, &llama_globals::qkv_weights,                 \
        &llama_globals::attn_norm_weights, &llama_globals::o_weights,         \
        &llama_globals::mlp_norm_weights, &llama_globals::up_weights,         \
        &llama_globals::gate_weights, &llama_globals::down_weights,           \
        &llama_globals::lm_head_norm_weights,                                 \
        &llama_globals::lm_head_weights, &llama_globals::k_cache,             \
        &llama_globals::v_cache, &llama_globals::rope_cos,                    \
        &llama_globals::rope_sin, &llama_globals::hidden_states,              \
        &llama_globals::q_post_rope, &llama_globals::attn_out,                \
        &llama_globals::attn_lse_intermediates,                               \
        &llama_globals::attn_out_intermediates,                               \
        &llama_globals::silu_out, &llama_globals::logits,                     \
        &llama_globals::pos_id, &llama_globals::attn_scale,                   \
        &llama_globals::rms_norm_eps, &llama_globals::skip_attn_reduction

#define BIND_FIXED(kernel, name, grid_blocks, block_threads)                  \
    bind_fixed_kernel<kernel, grid_blocks, block_threads>(                    \
        m, name, LLAMA_GLOBAL_MEMBERS)

#define BIND_FIXED_PDL(kernel, name, grid_blocks, block_threads)              \
    bind_fixed_kernel_pdl<kernel, grid_blocks, block_threads>(                \
        m, name, LLAMA_GLOBAL_MEMBERS)

#define BIND_FIXED_SMEM(kernel, name, grid_blocks, block_threads, smem_bytes) \
    bind_fixed_kernel_smem<kernel, grid_blocks, block_threads, smem_bytes>(    \
        m, name, LLAMA_GLOBAL_MEMBERS)

#define BIND_FIXED_SMEM_PDL(kernel, name, grid_blocks, block_threads,         \
                            smem_bytes)                                       \
    bind_fixed_kernel_smem_pdl<kernel, grid_blocks, block_threads,            \
                               smem_bytes>(m, name, LLAMA_GLOBAL_MEMBERS)

PYBIND11_MODULE(mk_mlp_simt, m) {
    m.doc() = "Low-resource batch-1 SIMT MLP controls for Hazy Megakernels";
    m.attr("cuda_compiler_version") =
        __CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 +
        __CUDACC_VER_BUILD__;
    m.attr("cuda_compiler_major") = __CUDACC_VER_MAJOR__;
    m.attr("cuda_compiler_minor") = __CUDACC_VER_MINOR__;
    m.attr("cuda_compiler_build") = __CUDACC_VER_BUILD__;
    m.attr("cuda_runtime_header_version") = CUDART_VERSION;
    m.def("begin_cuda_graph_capture", [](uintptr_t stream_ptr) {
        const cudaError_t status = cudaStreamBeginCapture(
            reinterpret_cast<cudaStream_t>(stream_ptr),
            cudaStreamCaptureModeGlobal);
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }
    });
    m.def("end_cuda_graph_capture", [](uintptr_t stream_ptr) {
        cudaGraph_t graph = nullptr;
        const cudaError_t status = cudaStreamEndCapture(
            reinterpret_cast<cudaStream_t>(stream_ptr), &graph);
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }
        return reinterpret_cast<uintptr_t>(graph);
    });
    m.def("destroy_cuda_graph", [](uintptr_t graph_ptr) {
        const cudaError_t status = cudaGraphDestroy(
            reinterpret_cast<cudaGraph_t>(graph_ptr));
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }
    });
    m.def("inspect_cuda_graph_edges", [](uintptr_t graph_ptr) {
        cudaGraph_t graph = reinterpret_cast<cudaGraph_t>(graph_ptr);
        size_t count = 0;
        cudaError_t status =
            graph_get_edges(graph, nullptr, nullptr, nullptr, &count);
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }
        std::vector<cudaGraphNode_t> from(count), to(count);
        std::vector<cudaGraphEdgeData> data(count);
        status = graph_get_edges(
            graph, from.data(), to.data(), data.data(), &count);
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }
        size_t kernel_kernel = 0;
        size_t default_edges = 0;
        size_t programmatic_edges = 0;
        size_t port_default = 0;
        size_t port_programmatic = 0;
        size_t port_launch_completion = 0;
        for (size_t index = 0; index < count; ++index) {
            cudaGraphNodeType from_type;
            cudaGraphNodeType to_type;
            cudaGraphNodeGetType(from[index], &from_type);
            cudaGraphNodeGetType(to[index], &to_type);
            kernel_kernel += from_type == cudaGraphNodeTypeKernel &&
                             to_type == cudaGraphNodeTypeKernel;
            default_edges +=
                data[index].type == cudaGraphDependencyTypeDefault;
            programmatic_edges +=
                data[index].type == cudaGraphDependencyTypeProgrammatic;
            port_default +=
                data[index].from_port == cudaGraphKernelNodePortDefault;
            port_programmatic += data[index].from_port ==
                                 cudaGraphKernelNodePortProgrammatic;
            port_launch_completion += data[index].from_port ==
                                      cudaGraphKernelNodePortLaunchCompletion;
        }
        pybind11::dict result;
        result["total"] = count;
        result["kernel_kernel"] = kernel_kernel;
        result["default_type"] = default_edges;
        result["programmatic_type"] = programmatic_edges;
        result["default_port"] = port_default;
        result["programmatic_port"] = port_programmatic;
        result["launch_completion_port"] = port_launch_completion;
        return result;
    });
    m.def("rewrite_cuda_graph_kernel_edges",
          [](uintptr_t graph_ptr, bool launch_completion,
             unsigned edge_slot_mask, unsigned edge_period,
             size_t max_edge_ordinals) {
              if (edge_period == 0 || edge_period > 31) {
                  throw std::runtime_error("invalid kernel edge period");
              }
              cudaGraph_t graph = reinterpret_cast<cudaGraph_t>(graph_ptr);
              size_t count = 0;
              cudaError_t status =
                  graph_get_edges(graph, nullptr, nullptr, nullptr, &count);
              if (status != cudaSuccess) {
                  throw std::runtime_error(cudaGetErrorString(status));
              }
              std::vector<cudaGraphNode_t> from(count), to(count);
              std::vector<cudaGraphEdgeData> old_data(count);
              status = graph_get_edges(
                  graph, from.data(), to.data(), old_data.data(), &count);
              if (status != cudaSuccess) {
                  throw std::runtime_error(cudaGetErrorString(status));
              }
              cudaGraphNode_t first_kernel = nullptr;
              for (size_t index = 0; index < count; ++index) {
                  cudaGraphNodeType from_type;
                  cudaGraphNodeType to_type;
                  cudaGraphNodeGetType(from[index], &from_type);
                  cudaGraphNodeGetType(to[index], &to_type);
                  if (from_type != cudaGraphNodeTypeKernel ||
                      to_type != cudaGraphNodeTypeKernel) {
                      continue;
                  }
                  bool has_kernel_predecessor = false;
                  for (size_t other = 0; other < count; ++other) {
                      cudaGraphNodeType other_from_type;
                      cudaGraphNodeType other_to_type;
                      cudaGraphNodeGetType(from[other], &other_from_type);
                      cudaGraphNodeGetType(to[other], &other_to_type);
                      if (other_from_type == cudaGraphNodeTypeKernel &&
                          other_to_type == cudaGraphNodeTypeKernel &&
                          to[other] == from[index]) {
                          has_kernel_predecessor = true;
                          break;
                      }
                  }
                  if (!has_kernel_predecessor) {
                      first_kernel = from[index];
                      break;
                  }
              }
              if (first_kernel == nullptr) {
                  throw std::runtime_error("no kernel chain found");
              }

              size_t rewritten = 0;
              size_t ordinal = 0;
              cudaGraphNode_t current = first_kernel;
              while (current != nullptr) {
                  if (max_edge_ordinals != 0 &&
                      ordinal >= max_edge_ordinals) {
                      break;
                  }
                  size_t index = count;
                  for (size_t candidate = 0; candidate < count; ++candidate) {
                      cudaGraphNodeType from_type;
                      cudaGraphNodeType to_type;
                      cudaGraphNodeGetType(from[candidate], &from_type);
                      cudaGraphNodeGetType(to[candidate], &to_type);
                      if (from_type == cudaGraphNodeTypeKernel &&
                          to_type == cudaGraphNodeTypeKernel &&
                          from[candidate] == current) {
                          index = candidate;
                          break;
                      }
                  }
                  if (index == count) {
                      break;
                  }
                  current = to[index];
                  const bool selected =
                      (edge_slot_mask & (1u << (ordinal % edge_period))) != 0;
                  ++ordinal;
                  if (!selected) {
                      continue;
                  }
                  status = graph_remove_dependencies(
                      graph, &from[index], &to[index], &old_data[index], 1);
                  if (status != cudaSuccess) {
                      throw std::runtime_error(cudaGetErrorString(status));
                  }
                  cudaGraphEdgeData replacement{};
                  replacement.type = cudaGraphDependencyTypeProgrammatic;
                  replacement.from_port =
                      launch_completion
                          ? cudaGraphKernelNodePortLaunchCompletion
                          : cudaGraphKernelNodePortProgrammatic;
                  status = graph_add_dependencies(
                      graph, &from[index], &to[index], &replacement, 1);
                  if (status != cudaSuccess) {
                      throw std::runtime_error(cudaGetErrorString(status));
                  }
                  ++rewritten;
              }
              return rewritten;
          },
          pybind11::arg("graph_ptr"),
          pybind11::arg("launch_completion"),
          pybind11::arg("edge_slot_mask"),
          pybind11::arg("edge_period"),
          pybind11::arg("max_edge_ordinals") = 0);
    m.def("set_cuda_graph_kernel_chain_priorities",
          [](uintptr_t graph_ptr, pybind11::list priority_pattern) {
              if (priority_pattern.size() == 0) {
                  throw std::runtime_error("priority pattern must be non-empty");
              }
              std::vector<int> priorities;
              priorities.reserve(priority_pattern.size());
              for (const auto value : priority_pattern) {
                  priorities.push_back(pybind11::cast<int>(value));
              }

              int least_priority = 0;
              int greatest_priority = 0;
              cudaError_t status = cudaDeviceGetStreamPriorityRange(
                  &least_priority, &greatest_priority);
              if (status != cudaSuccess) {
                  throw std::runtime_error(cudaGetErrorString(status));
              }
              for (const int priority : priorities) {
                  if (priority < greatest_priority ||
                      priority > least_priority) {
                      throw std::runtime_error(
                          "kernel-node priority is outside the device range");
                  }
              }

              cudaGraph_t graph = reinterpret_cast<cudaGraph_t>(graph_ptr);
              size_t count = 0;
              status = graph_get_edges(
                  graph, nullptr, nullptr, nullptr, &count);
              if (status != cudaSuccess) {
                  throw std::runtime_error(cudaGetErrorString(status));
              }
              std::vector<cudaGraphNode_t> from(count), to(count);
              std::vector<cudaGraphEdgeData> edge_data(count);
              status = graph_get_edges(
                  graph, from.data(), to.data(), edge_data.data(), &count);
              if (status != cudaSuccess) {
                  throw std::runtime_error(cudaGetErrorString(status));
              }

              cudaGraphNode_t first_kernel = nullptr;
              for (size_t index = 0; index < count; ++index) {
                  cudaGraphNodeType from_type;
                  cudaGraphNodeType to_type;
                  cudaGraphNodeGetType(from[index], &from_type);
                  cudaGraphNodeGetType(to[index], &to_type);
                  if (from_type != cudaGraphNodeTypeKernel ||
                      to_type != cudaGraphNodeTypeKernel) {
                      continue;
                  }
                  bool has_kernel_predecessor = false;
                  for (size_t other = 0; other < count; ++other) {
                      cudaGraphNodeType other_from_type;
                      cudaGraphNodeType other_to_type;
                      cudaGraphNodeGetType(from[other], &other_from_type);
                      cudaGraphNodeGetType(to[other], &other_to_type);
                      if (other_from_type == cudaGraphNodeTypeKernel &&
                          other_to_type == cudaGraphNodeTypeKernel &&
                          to[other] == from[index]) {
                          has_kernel_predecessor = true;
                          break;
                      }
                  }
                  if (!has_kernel_predecessor) {
                      first_kernel = from[index];
                      break;
                  }
              }
              if (first_kernel == nullptr) {
                  throw std::runtime_error("no kernel chain found");
              }

              size_t ordinal = 0;
              cudaGraphNode_t current = first_kernel;
              while (current != nullptr) {
                  cudaKernelNodeAttrValue value{};
                  value.priority = priorities[ordinal % priorities.size()];
                  status = cudaGraphKernelNodeSetAttribute(
                      current, cudaKernelNodeAttributePriority, &value);
                  if (status != cudaSuccess) {
                      throw std::runtime_error(cudaGetErrorString(status));
                  }
                  ++ordinal;

                  cudaGraphNode_t next = nullptr;
                  for (size_t index = 0; index < count; ++index) {
                      cudaGraphNodeType from_type;
                      cudaGraphNodeType to_type;
                      cudaGraphNodeGetType(from[index], &from_type);
                      cudaGraphNodeGetType(to[index], &to_type);
                      if (from_type == cudaGraphNodeTypeKernel &&
                          to_type == cudaGraphNodeTypeKernel &&
                          from[index] == current) {
                          next = to[index];
                          break;
                      }
                  }
                  current = next;
              }

              pybind11::dict result;
              result["kernel_nodes"] = ordinal;
              result["least_priority"] = least_priority;
              result["greatest_priority"] = greatest_priority;
              return result;
          });
    m.def("instantiate_cuda_graph_exec",
          [](uintptr_t graph_ptr, unsigned long long flags) {
              cudaGraphExec_t executable = nullptr;
              const cudaError_t status = cudaGraphInstantiateWithFlags(
                  &executable, reinterpret_cast<cudaGraph_t>(graph_ptr),
                  flags);
              if (status != cudaSuccess) {
                  throw std::runtime_error(cudaGetErrorString(status));
              }
              return reinterpret_cast<uintptr_t>(executable);
          });
    m.def("launch_cuda_graph_exec",
          [](uintptr_t executable_ptr, uintptr_t stream_ptr) {
              const cudaError_t status = cudaGraphLaunch(
                  reinterpret_cast<cudaGraphExec_t>(executable_ptr),
                  reinterpret_cast<cudaStream_t>(stream_ptr));
              if (status != cudaSuccess) {
                  throw std::runtime_error(cudaGetErrorString(status));
              }
          });
    m.def("destroy_cuda_graph_exec", [](uintptr_t executable_ptr) {
        const cudaError_t status = cudaGraphExecDestroy(
            reinterpret_cast<cudaGraphExec_t>(executable_ptr));
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }
    });
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_8w4r, false>),
               "opcode5_simt", megakernel::mlp_simt::upgate_8w4r::ctas,
               megakernel::mlp_simt::upgate_8w4r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_4w8r, false>),
               "opcode5_simt_4w8r", megakernel::mlp_simt::upgate_4w8r::ctas,
               megakernel::mlp_simt::upgate_4w8r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_8w8r, false>),
               "opcode5_simt_8w8r", megakernel::mlp_simt::upgate_8w8r::ctas,
               megakernel::mlp_simt::upgate_8w8r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_8w8r, true>),
               "opcode5_simt_8w8r_i",
               megakernel::mlp_simt::upgate_8w8r::ctas,
               megakernel::mlp_simt::upgate_8w8r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_8w8r, true, -2>),
               "opcode5_simt_8w8r_i_cs",
               megakernel::mlp_simt::upgate_8w8r::ctas,
               megakernel::mlp_simt::upgate_8w8r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_16w4r, false>),
               "opcode5_simt_16w4r",
               megakernel::mlp_simt::upgate_16w4r::ctas,
               megakernel::mlp_simt::upgate_16w4r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_8w4r, true>),
               "opcode5_simt_8w4r_i",
               megakernel::mlp_simt::upgate_8w4r::ctas,
               megakernel::mlp_simt::upgate_8w4r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_16w2r, true>),
               "opcode5_simt_16w2r_i",
               megakernel::mlp_simt::upgate_16w2r::ctas,
               megakernel::mlp_simt::upgate_16w2r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_16w4r, true>),
               "opcode5_simt_16w4r_i",
               megakernel::mlp_simt::upgate_16w4r::ctas,
               megakernel::mlp_simt::upgate_16w4r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_32w1r, true>),
               "opcode5_simt_32w1r_i",
               megakernel::mlp_simt::upgate_32w1r::ctas,
               megakernel::mlp_simt::upgate_32w1r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_32w2r, true>),
               "opcode5_simt_32w2r_i",
               megakernel::mlp_simt::upgate_32w2r::ctas,
               megakernel::mlp_simt::upgate_32w2r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_32w1r_264, true>),
               "opcode5_simt_32w1r_i_264cta",
               megakernel::mlp_simt::upgate_32w1r_264::ctas,
               megakernel::mlp_simt::upgate_32w1r_264::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_32w2r_132, true>),
               "opcode5_simt_32w2r_i_132cta",
               megakernel::mlp_simt::upgate_32w2r_132::ctas,
               megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_32w2r, true, 128>),
               "opcode5_simt_32w2r_i_l2_128b",
               megakernel::mlp_simt::upgate_32w2r::ctas,
               megakernel::mlp_simt::upgate_32w2r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_32w2r, true, 256>),
               "opcode5_simt_32w2r_i_l2_256b",
               megakernel::mlp_simt::upgate_32w2r::ctas,
               megakernel::mlp_simt::upgate_32w2r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate_v4<
                   megakernel::mlp_simt::upgate_32w2r>),
               "opcode5_simt_32w2r_i_v4",
               megakernel::mlp_simt::upgate_32w2r::ctas,
               megakernel::mlp_simt::upgate_32w2r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate_132_pairwarp<
                   megakernel::mlp_simt::upgate_32w2r_132>),
               "opcode5_simt_32w2r_i_132cta_pairwarp",
               megakernel::mlp_simt::upgate_32w2r_132::ctas,
               megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate_132_pairwarp<
                   megakernel::mlp_simt::upgate_32w2r_132, -3, true>),
               "opcode5_simt_32w2r_i_132cta_allwarp_na",
               megakernel::mlp_simt::upgate_32w2r_132::ctas,
               megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate_132_pairwarp<
                   megakernel::mlp_simt::upgate_32w2r_132, -5, true>),
               "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b",
               megakernel::mlp_simt::upgate_32w2r_132::ctas,
               megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate_132_pairwarp<
                   megakernel::mlp_simt::upgate_32w2r_132, -5, true,
                   false, false, -1, 0, 3, 0, false,
                   megakernel::mlp_simt::prefetch_policy_l2, true>),
               "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_v4",
               megakernel::mlp_simt::upgate_32w2r_132::ctas,
               megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate_132_pairwarp<
                   megakernel::mlp_simt::upgate_32w2r_132, -5, true, true>),
               "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_trigger",
               megakernel::mlp_simt::upgate_32w2r_132::ctas,
               megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_pairwarp<
                        megakernel::mlp_simt::upgate_32w2r_132, -5, true,
                        true>),
                   "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"
                   "unsafe_no_wait_trigger",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
#define BIND_OP5_TRIGGER_PAIR(pair_value, pair_name)                          \
    BIND_FIXED((megakernel::mlp_simt::upgate_132_pairwarp<                   \
                   megakernel::mlp_simt::upgate_32w2r_132, -5, true, true,   \
                   false, pair_value>),                                      \
               "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_trigger_" \
               pair_name,                                                    \
               megakernel::mlp_simt::upgate_32w2r_132::ctas,                 \
               megakernel::mlp_simt::upgate_32w2r_132::threads);             \
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_pairwarp<               \
                        megakernel::mlp_simt::upgate_32w2r_132, -5, true,    \
                        true, true, pair_value>),                             \
                   "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"     \
                   "wait_trigger_" pair_name,                               \
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,             \
                   megakernel::mlp_simt::upgate_32w2r_132::threads)
    BIND_OP5_TRIGGER_PAIR(512, "k50");
    BIND_OP5_TRIGGER_PAIR(768, "k75");
    BIND_OP5_TRIGGER_PAIR(896, "k875");
    BIND_OP5_TRIGGER_PAIR(960, "k9375");
    BIND_OP5_TRIGGER_PAIR(1024, "epilogue");
#undef BIND_OP5_TRIGGER_PAIR
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_pairwarp<
                        megakernel::mlp_simt::upgate_32w2r_132, -5, true, true,
                        true>),
                   "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_wait_trigger",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_down_fused_132<false>),
                   "opcode56_simt_fused_132cta_pdl_wait_trigger",
                   132, 1024);
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_down_fused_132<true>),
                   "opcode56_simt_fused_132cta_pdl_wait_trigger_epilogue",
                   132, 1024);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_32w2r, true, -1>),
               "opcode5_simt_32w2r_i_cg",
               megakernel::mlp_simt::upgate_32w2r::ctas,
               megakernel::mlp_simt::upgate_32w2r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_32w2r, true, -2>),
               "opcode5_simt_32w2r_i_cs",
               megakernel::mlp_simt::upgate_32w2r::ctas,
               megakernel::mlp_simt::upgate_32w2r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_32w2r, true, -3>),
               "opcode5_simt_32w2r_i_na",
               megakernel::mlp_simt::upgate_32w2r::ctas,
               megakernel::mlp_simt::upgate_32w2r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_32w2r, true, -4>),
               "opcode5_simt_32w2r_i_na_l2_128b",
               megakernel::mlp_simt::upgate_32w2r::ctas,
               megakernel::mlp_simt::upgate_32w2r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_32w2r, true, -5>),
               "opcode5_simt_32w2r_i_na_l2_256b",
               megakernel::mlp_simt::upgate_32w2r::ctas,
               megakernel::mlp_simt::upgate_32w2r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_32w2r, true, -3, true>),
               "opcode5_simt_32w2r_i_adjacent_na",
               megakernel::mlp_simt::upgate_32w2r::ctas,
               megakernel::mlp_simt::upgate_32w2r::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_32w2r_132, true, -3>),
               "opcode5_simt_32w2r_i_132cta_na",
               megakernel::mlp_simt::upgate_32w2r_132::ctas,
               megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED((megakernel::mlp_simt::upgate<
                   megakernel::mlp_simt::upgate_32w2r_132, true, -5>),
               "opcode5_simt_32w2r_i_132cta_na_l2_256b",
               megakernel::mlp_simt::upgate_32w2r_132::ctas,
               megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED_SMEM((megakernel::mlp_simt::upgate_cp_async<
                         megakernel::mlp_simt::upgate_32w2r, 64>),
                    "opcode5_simt_32w2r_cp64",
                    megakernel::mlp_simt::upgate_32w2r::ctas,
                    megakernel::mlp_simt::upgate_32w2r::threads,
                    2 * 32 * 4 * 64 * sizeof(megakernel::mlp_simt::bf16));
    BIND_FIXED_SMEM((megakernel::mlp_simt::upgate_cp_async<
                         megakernel::mlp_simt::upgate_32w2r, 128>),
                    "opcode5_simt_32w2r_cp128",
                    megakernel::mlp_simt::upgate_32w2r::ctas,
                    megakernel::mlp_simt::upgate_32w2r::threads,
                    2 * 32 * 4 * 128 * sizeof(megakernel::mlp_simt::bf16));
    BIND_FIXED_SMEM((megakernel::mlp_simt::upgate_cp_async<
                         megakernel::mlp_simt::upgate_32w2r, 256>),
                    "opcode5_simt_32w2r_cp256",
                    megakernel::mlp_simt::upgate_32w2r::ctas,
                    megakernel::mlp_simt::upgate_32w2r::threads,
                    2 * 32 * 4 * 256 * sizeof(megakernel::mlp_simt::bf16));
    BIND_FIXED_SMEM((megakernel::mlp_simt::upgate_cp_async<
                         megakernel::mlp_simt::upgate_32w2r, 64, 3>),
                    "opcode5_simt_32w2r_cp64x3",
                    megakernel::mlp_simt::upgate_32w2r::ctas,
                    megakernel::mlp_simt::upgate_32w2r::threads,
                    3 * 32 * 4 * 64 * sizeof(megakernel::mlp_simt::bf16));
    BIND_FIXED_SMEM((megakernel::mlp_simt::upgate_cp_async<
                         megakernel::mlp_simt::upgate_32w2r, 128, 3>),
                    "opcode5_simt_32w2r_cp128x3",
                    megakernel::mlp_simt::upgate_32w2r::ctas,
                    megakernel::mlp_simt::upgate_32w2r::threads,
                    3 * 32 * 4 * 128 * sizeof(megakernel::mlp_simt::bf16));
    BIND_FIXED_SMEM_PDL((megakernel::mlp_simt::upgate_cp_async<
                             megakernel::mlp_simt::upgate_32w2r, 128, 3,
                             true, true>),
                        "opcode5_simt_32w2r_cp128x3_pdl_wait_trigger",
                        megakernel::mlp_simt::upgate_32w2r::ctas,
                        megakernel::mlp_simt::upgate_32w2r::threads,
                        3 * 32 * 4 * 128 *
                            sizeof(megakernel::mlp_simt::bf16));
    BIND_FIXED_SMEM_PDL((megakernel::mlp_simt::upgate_cp_async<
                             megakernel::mlp_simt::upgate_32w2r, 256, 2,
                             true, true>),
                        "opcode5_simt_32w2r_cp256x2_pdl_wait_trigger",
                        megakernel::mlp_simt::upgate_32w2r::ctas,
                        megakernel::mlp_simt::upgate_32w2r::threads,
                        2 * 32 * 4 * 256 *
                            sizeof(megakernel::mlp_simt::bf16));
    BIND_FIXED_SMEM_PDL((megakernel::mlp_simt::upgate_cp_async<
                             megakernel::mlp_simt::upgate_32w2r, 256, 3,
                             true, true>),
                        "opcode5_simt_32w2r_cp256x3_pdl_wait_trigger",
                        megakernel::mlp_simt::upgate_32w2r::ctas,
                        megakernel::mlp_simt::upgate_32w2r::threads,
                        3 * 32 * 4 * 256 *
                            sizeof(megakernel::mlp_simt::bf16));

    BIND_FIXED((megakernel::mlp_simt::downproj<
                   megakernel::mlp_simt::down_8w2r>),
               "opcode6_simt", megakernel::mlp_simt::down_8w2r::ctas,
               megakernel::mlp_simt::down_8w2r::threads);
    BIND_FIXED((megakernel::mlp_simt::downproj<
                   megakernel::mlp_simt::down_4w4r>),
               "opcode6_simt_4w4r", megakernel::mlp_simt::down_4w4r::ctas,
               megakernel::mlp_simt::down_4w4r::threads);
    BIND_FIXED((megakernel::mlp_simt::downproj<
                   megakernel::mlp_simt::down_8w4r>),
               "opcode6_simt_8w4r", megakernel::mlp_simt::down_8w4r::ctas,
               megakernel::mlp_simt::down_8w4r::threads);
    BIND_FIXED((megakernel::mlp_simt::downproj<
                   megakernel::mlp_simt::down_16w1r>),
               "opcode6_simt_16w1r", megakernel::mlp_simt::down_16w1r::ctas,
               megakernel::mlp_simt::down_16w1r::threads);
    BIND_FIXED((megakernel::mlp_simt::downproj_cached<
                   megakernel::mlp_simt::down_8w2r>),
               "opcode6_simt_cached", megakernel::mlp_simt::down_8w2r::ctas,
               megakernel::mlp_simt::down_8w2r::threads);
    BIND_FIXED((megakernel::mlp_simt::downproj_v4<
                   megakernel::mlp_simt::down_16w1r, 0>),
               "opcode6_simt_16w1r_v4", megakernel::mlp_simt::down_16w1r::ctas,
               megakernel::mlp_simt::down_16w1r::threads);
    BIND_FIXED((megakernel::mlp_simt::downproj_v4<
                   megakernel::mlp_simt::down_16w1r, -3>),
               "opcode6_simt_16w1r_v4_na",
               megakernel::mlp_simt::down_16w1r::ctas,
               megakernel::mlp_simt::down_16w1r::threads);
    BIND_FIXED((megakernel::mlp_simt::downproj_v4<
                   megakernel::mlp_simt::down_16w1r, -5>),
               "opcode6_simt_16w1r_v4_na_l2_256b",
               megakernel::mlp_simt::down_16w1r::ctas,
               megakernel::mlp_simt::down_16w1r::threads);
    BIND_FIXED((megakernel::mlp_simt::downproj_v4<
                   megakernel::mlp_simt::down_16w1r, -3, 32>),
               "opcode6_simt_16w1r_v4_na_prefetch4k",
               megakernel::mlp_simt::down_16w1r::ctas,
               megakernel::mlp_simt::down_16w1r::threads);
    BIND_FIXED((megakernel::mlp_simt::downproj_v4<
                   megakernel::mlp_simt::down_16w1r, -3, 64>),
               "opcode6_simt_16w1r_v4_na_prefetch8k",
               megakernel::mlp_simt::down_16w1r::ctas,
               megakernel::mlp_simt::down_16w1r::threads);
    BIND_FIXED((megakernel::mlp_simt::downproj_v4<
                   megakernel::mlp_simt::down_16w1r, -3, 128>),
               "opcode6_simt_16w1r_v4_na_prefetch16k",
               megakernel::mlp_simt::down_16w1r::ctas,
               megakernel::mlp_simt::down_16w1r::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::downproj_v4<
                        megakernel::mlp_simt::down_16w1r, -3, 0, true>),
                   "opcode6_simt_16w1r_v4_na_pdl_wait",
                   megakernel::mlp_simt::down_16w1r::ctas,
                   megakernel::mlp_simt::down_16w1r::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::downproj_v4<
                        megakernel::mlp_simt::down_16w1r, -3>),
                   "opcode6_simt_16w1r_v4_na_pdl_unsafe_no_wait",
                   megakernel::mlp_simt::down_16w1r::ctas,
                   megakernel::mlp_simt::down_16w1r::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::downproj_v4<
                        megakernel::mlp_simt::down_16w1r, -3, 0, true, true>),
                   "opcode6_simt_16w1r_v4_na_pdl_wait_trigger",
                   megakernel::mlp_simt::down_16w1r::ctas,
                   megakernel::mlp_simt::down_16w1r::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::downproj_v4<
                        megakernel::mlp_simt::down_16w1r, -3, 8, true, true,
                        -1, megakernel::mlp_simt::prefetch_policy_l1>),
                   "opcode6_simt_16w1r_v4_na_pdl_prefetch1k_l1_trigger",
                   megakernel::mlp_simt::down_16w1r::ctas,
                   megakernel::mlp_simt::down_16w1r::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::downproj_v4<
                        megakernel::mlp_simt::down_16w1r, -3, 8, true, true,
                        -1,
                        megakernel::mlp_simt::prefetch_policy_l2_evict_last>),
                   "opcode6_simt_16w1r_v4_na_pdl_prefetch1k_l2_evict_last_"
                   "trigger",
                   megakernel::mlp_simt::down_16w1r::ctas,
                   megakernel::mlp_simt::down_16w1r::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::downproj_split4_pdl),
                   "opcode6_simt_split4_pdl_trigger", 132, 512);
#define BIND_OP6_SMEM_PREFIX(cols_value, bytes_name)                         \
    BIND_FIXED_SMEM_PDL((megakernel::mlp_simt::downproj_smem_prefix_pdl<    \
                             megakernel::mlp_simt::down_16w1r, cols_value>),\
                        "opcode6_simt_16w1r_smem_prefix" bytes_name        \
                        "_pdl_wait_trigger",                               \
                        megakernel::mlp_simt::down_16w1r::ctas,             \
                        megakernel::mlp_simt::down_16w1r::threads,          \
                        16 * cols_value *                                   \
                            sizeof(megakernel::mlp_simt::bf16))
    BIND_OP6_SMEM_PREFIX(256, "512b");
    BIND_OP6_SMEM_PREFIX(512, "1k");
    BIND_OP6_SMEM_PREFIX(1024, "2k");
    BIND_OP6_SMEM_PREFIX(2048, "4k");
#undef BIND_OP6_SMEM_PREFIX
#define BIND_OP6_TRIGGER_QUAD(quad_value, quad_name)                          \
    BIND_FIXED_PDL((megakernel::mlp_simt::downproj_v4<                       \
                        megakernel::mlp_simt::down_16w1r, -3, 0, true, true, \
                        quad_value>),                                         \
                   "opcode6_simt_16w1r_v4_na_pdl_wait_trigger_" quad_name,   \
                   megakernel::mlp_simt::down_16w1r::ctas,                   \
                   megakernel::mlp_simt::down_16w1r::threads)
    BIND_OP6_TRIGGER_QUAD(1024, "k50");
    BIND_OP6_TRIGGER_QUAD(1536, "k75");
    BIND_OP6_TRIGGER_QUAD(1792, "k875");
    BIND_OP6_TRIGGER_QUAD(1920, "k9375");
    BIND_OP6_TRIGGER_QUAD(2048, "epilogue");
#undef BIND_OP6_TRIGGER_QUAD
#define BIND_OP6_PDL_PREFETCH(lines_value, bytes_name)                        \
    BIND_FIXED_PDL((megakernel::mlp_simt::downproj_v4<                       \
                        megakernel::mlp_simt::down_16w1r, -3, lines_value,   \
                        true>),                                               \
                   "opcode6_simt_16w1r_v4_na_pdl_prefetch" bytes_name,       \
                   megakernel::mlp_simt::down_16w1r::ctas,                   \
                   megakernel::mlp_simt::down_16w1r::threads)
    BIND_OP6_PDL_PREFETCH(1, "128b");
    BIND_OP6_PDL_PREFETCH(2, "256b");
    BIND_OP6_PDL_PREFETCH(4, "512b");
    BIND_OP6_PDL_PREFETCH(8, "1k");
    BIND_OP6_PDL_PREFETCH(16, "2k");
#undef BIND_OP6_PDL_PREFETCH
#define BIND_OP6_PDL_PREFETCH_TRIGGER(lines_value, bytes_name)               \
    BIND_FIXED_PDL((megakernel::mlp_simt::downproj_v4<                       \
                        megakernel::mlp_simt::down_16w1r, -3, lines_value,   \
                        true, true>),                                         \
                   "opcode6_simt_16w1r_v4_na_pdl_prefetch" bytes_name      \
                   "_trigger",                                              \
                   megakernel::mlp_simt::down_16w1r::ctas,                   \
                   megakernel::mlp_simt::down_16w1r::threads)
    BIND_OP6_PDL_PREFETCH_TRIGGER(1, "128b");
    BIND_OP6_PDL_PREFETCH_TRIGGER(2, "256b");
    BIND_OP6_PDL_PREFETCH_TRIGGER(4, "512b");
    BIND_OP6_PDL_PREFETCH_TRIGGER(8, "1k");
    BIND_OP6_PDL_PREFETCH_TRIGGER(16, "2k");
#undef BIND_OP6_PDL_PREFETCH_TRIGGER
#define BIND_OP6_PREFETCH1K_TRIGGER_QUAD(quad_value, quad_name)              \
    BIND_FIXED_PDL((megakernel::mlp_simt::downproj_v4<                      \
                        megakernel::mlp_simt::down_16w1r, -3, 8, true,      \
                        true, quad_value>),                                  \
                   "opcode6_simt_16w1r_v4_na_pdl_prefetch1k_trigger_"      \
                   quad_name,                                               \
                   megakernel::mlp_simt::down_16w1r::ctas,                  \
                   megakernel::mlp_simt::down_16w1r::threads)
    BIND_OP6_PREFETCH1K_TRIGGER_QUAD(1024, "k50");
    BIND_OP6_PREFETCH1K_TRIGGER_QUAD(1536, "k75");
    BIND_OP6_PREFETCH1K_TRIGGER_QUAD(1792, "k875");
    BIND_OP6_PREFETCH1K_TRIGGER_QUAD(1920, "k9375");
    BIND_OP6_PREFETCH1K_TRIGGER_QUAD(2048, "epilogue");
#undef BIND_OP6_PREFETCH1K_TRIGGER_QUAD
    BIND_FIXED_PDL((megakernel::mlp_simt::downproj_v4<
                        megakernel::mlp_simt::down_16w1r, -3, 32, true>),
                   "opcode6_simt_16w1r_v4_na_pdl_prefetch4k",
                   megakernel::mlp_simt::down_16w1r::ctas,
                   megakernel::mlp_simt::down_16w1r::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::downproj_v4<
                        megakernel::mlp_simt::down_16w1r, -3, 64, true>),
                   "opcode6_simt_16w1r_v4_na_pdl_prefetch8k",
                   megakernel::mlp_simt::down_16w1r::ctas,
                   megakernel::mlp_simt::down_16w1r::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::downproj_v4<
                        megakernel::mlp_simt::down_16w1r, -3, 128, true>),
                   "opcode6_simt_16w1r_v4_na_pdl_prefetch16k",
                   megakernel::mlp_simt::down_16w1r::ctas,
                   megakernel::mlp_simt::down_16w1r::threads);
    BIND_FIXED_SMEM((megakernel::mlp_simt::downproj_cp_async<
                         megakernel::mlp_simt::down_16w1r, 256>),
                    "opcode6_simt_16w1r_cp256",
                    megakernel::mlp_simt::down_16w1r::ctas,
                    megakernel::mlp_simt::down_16w1r::threads,
                    2 * 16 * 256 * sizeof(megakernel::mlp_simt::bf16));
    BIND_FIXED_SMEM((megakernel::mlp_simt::downproj_cp_async<
                         megakernel::mlp_simt::down_16w1r, 512>),
                    "opcode6_simt_16w1r_cp512",
                    megakernel::mlp_simt::down_16w1r::ctas,
                    megakernel::mlp_simt::down_16w1r::threads,
                    2 * 16 * 512 * sizeof(megakernel::mlp_simt::bf16));
    BIND_FIXED_SMEM((megakernel::mlp_simt::downproj_cp_async<
                         megakernel::mlp_simt::down_16w1r, 1024>),
                    "opcode6_simt_16w1r_cp1024",
                    megakernel::mlp_simt::down_16w1r::ctas,
                    megakernel::mlp_simt::down_16w1r::threads,
                    2 * 16 * 1024 * sizeof(megakernel::mlp_simt::bf16));
    BIND_FIXED_SMEM((megakernel::mlp_simt::downproj_cp_async<
                         megakernel::mlp_simt::down_16w1r, 512, 3>),
                    "opcode6_simt_16w1r_cp512x3",
                    megakernel::mlp_simt::down_16w1r::ctas,
                    megakernel::mlp_simt::down_16w1r::threads,
                    3 * 16 * 512 * sizeof(megakernel::mlp_simt::bf16));
    BIND_FIXED_SMEM((megakernel::mlp_simt::downproj_cp_async<
                         megakernel::mlp_simt::down_16w1r, 1024, 3>),
                    "opcode6_simt_16w1r_cp1024x3",
                    megakernel::mlp_simt::down_16w1r::ctas,
                    megakernel::mlp_simt::down_16w1r::threads,
                    3 * 16 * 1024 * sizeof(megakernel::mlp_simt::bf16));

    BIND_FIXED((megakernel::qkv_simt::qkv_rope_append<
                   megakernel::qkv_simt::qkv_12w_default>),
               "opcode1_simt_12w", megakernel::qkv_simt::grid_ctas,
               megakernel::qkv_simt::qkv_12w_default::threads);
    BIND_FIXED((megakernel::qkv_simt::qkv_rope_append<
                   megakernel::qkv_simt::qkv_12w_stream>),
               "opcode1_simt_12w_stream", megakernel::qkv_simt::grid_ctas,
               megakernel::qkv_simt::qkv_12w_stream::threads);
    BIND_FIXED((megakernel::qkv_simt::qkv_rope_append<
                   megakernel::qkv_simt::qkv_24w_default>),
               "opcode1_simt_24w", megakernel::qkv_simt::grid_ctas,
               megakernel::qkv_simt::qkv_24w_default::threads);
    BIND_FIXED((megakernel::qkv_simt::qkv_rope_append<
                   megakernel::qkv_simt::qkv_24w_default, false, true>),
               "opcode1_simt_24w_pdl_trigger",
               megakernel::qkv_simt::grid_ctas,
               megakernel::qkv_simt::qkv_24w_default::threads);
    BIND_FIXED((megakernel::qkv_simt::qkv_rope_append<
                   megakernel::qkv_simt::qkv_24w_default, false, true, 0,
                   -1, false, false, false,
                   megakernel::mlp_simt::prefetch_policy_l2, true>),
               "opcode1_simt_24w_pdl_trigger_qready",
               megakernel::qkv_simt::grid_ctas,
               megakernel::qkv_simt::qkv_24w_default::threads);
    BIND_FIXED((megakernel::qkv_simt::qkv_rope_append<
                   megakernel::qkv_simt::qkv_24w_default, false, true, 0,
                   -1, false, true>),
               "opcode1_simt_24w_pdl_trigger_fused_context1_attention",
               megakernel::qkv_simt::grid_ctas,
               megakernel::qkv_simt::qkv_24w_default::threads);
    BIND_FIXED((megakernel::qkv_simt::qkv_rope_append<
                   megakernel::qkv_simt::qkv_8w_default, false, true, 0,
                   -1, false, true, true>),
               "opcode1_simt_8w_pdl_trigger_fused_context1_attention_"
               "kv_only",
               megakernel::qkv_simt::kv_only_grid_ctas,
               megakernel::qkv_simt::qkv_8w_default::threads);
    BIND_FIXED((megakernel::qkv_simt::qkv_rope_append<
                   megakernel::qkv_simt::qkv_24w_default, false, true, 0,
                   -1, true>),
               "opcode1_simt_24w_pdl_trigger_vready",
               megakernel::qkv_simt::grid_ctas,
               megakernel::qkv_simt::qkv_24w_default::threads);
#define BIND_QKV_TRIGGER_PAIR(pair_value, pair_name)                         \
    BIND_FIXED((megakernel::qkv_simt::qkv_rope_append<                      \
                   megakernel::qkv_simt::qkv_24w_default, false, true, 0,  \
                   pair_value>),                                            \
               "opcode1_simt_24w_pdl_trigger_" pair_name,                  \
               megakernel::qkv_simt::grid_ctas,                            \
               megakernel::qkv_simt::qkv_24w_default::threads)
    BIND_QKV_TRIGGER_PAIR(0, "mainloop");
    BIND_QKV_TRIGGER_PAIR(512, "k50");
    BIND_QKV_TRIGGER_PAIR(768, "k75");
    BIND_QKV_TRIGGER_PAIR(896, "k875");
    BIND_QKV_TRIGGER_PAIR(960, "k9375");
    BIND_QKV_TRIGGER_PAIR(1024, "epilogue");
#undef BIND_QKV_TRIGGER_PAIR
#define BIND_QKV_TRIGGER_TAIL(tail_value, tail_name)                        \
    BIND_FIXED((megakernel::qkv_simt::qkv_rope_append<                     \
                   megakernel::qkv_simt::qkv_24w_default, false, true, 0, \
                   -1, false, false, false,                               \
                   megakernel::mlp_simt::prefetch_policy_l2, false,       \
                   tail_value>),                                          \
               "opcode1_simt_24w_pdl_trigger_" tail_name,                \
               megakernel::qkv_simt::grid_ctas,                           \
               megakernel::qkv_simt::qkv_24w_default::threads)
    BIND_QKV_TRIGGER_TAIL(1, "tail1");
    BIND_QKV_TRIGGER_TAIL(2, "tail2");
    BIND_QKV_TRIGGER_TAIL(4, "tail4");
#undef BIND_QKV_TRIGGER_TAIL
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<
                        megakernel::qkv_simt::qkv_24w_default, true>),
                   "opcode1_simt_24w_pdl_wait",
                   megakernel::qkv_simt::grid_ctas,
                   megakernel::qkv_simt::qkv_24w_default::threads);
#define BIND_QKV_WAIT_PREFETCH(lines_value, bytes_name)                       \
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<                   \
                        megakernel::qkv_simt::qkv_24w_default, true, false,   \
                        lines_value>),                                        \
                   "opcode1_simt_24w_pdl_wait_prefetch" bytes_name,          \
                   megakernel::qkv_simt::grid_ctas,                          \
                   megakernel::qkv_simt::qkv_24w_default::threads)
    BIND_QKV_WAIT_PREFETCH(1, "128");
    BIND_QKV_WAIT_PREFETCH(2, "256");
    BIND_QKV_WAIT_PREFETCH(4, "512");
    BIND_QKV_WAIT_PREFETCH(8, "1024");
#undef BIND_QKV_WAIT_PREFETCH
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<
                        megakernel::qkv_simt::qkv_24w_default, true, true>),
                   "opcode1_simt_24w_pdl_wait_trigger",
                   megakernel::qkv_simt::grid_ctas,
                   megakernel::qkv_simt::qkv_24w_default::threads);
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<
                        megakernel::qkv_simt::qkv_24w_default, true, true, 2,
                        -1, false, false, false,
                        megakernel::mlp_simt::prefetch_policy_l2, true>),
                   "opcode1_simt_24w_pdl_wait_trigger_prefetch256_qready",
                   megakernel::qkv_simt::grid_ctas,
                   megakernel::qkv_simt::qkv_24w_default::threads);
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<
                        megakernel::qkv_simt::qkv_24w_default, true, true, 0,
                        -1, true>),
                   "opcode1_simt_24w_pdl_wait_trigger_vready",
                   megakernel::qkv_simt::grid_ctas,
                   megakernel::qkv_simt::qkv_24w_default::threads);
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<
                        megakernel::qkv_simt::qkv_24w_default, true, true, 1>),
                   "opcode1_simt_24w_pdl_wait_trigger_prefetch128",
                   megakernel::qkv_simt::grid_ctas,
                   megakernel::qkv_simt::qkv_24w_default::threads);
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<
                        megakernel::qkv_simt::qkv_24w_default, true, true, 1,
                        -1, false, true>),
                   "opcode1_simt_24w_pdl_wait_trigger_prefetch128_"
                   "fused_context1_attention",
                   megakernel::qkv_simt::grid_ctas,
                   megakernel::qkv_simt::qkv_24w_default::threads);
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<
                        megakernel::qkv_simt::qkv_8w_default, true, true, 1,
                        -1, false, true, true>),
                   "opcode1_simt_8w_pdl_wait_trigger_prefetch128_"
                   "fused_context1_attention_kv_only",
                   megakernel::qkv_simt::kv_only_grid_ctas,
                   megakernel::qkv_simt::qkv_8w_default::threads);
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<
                        megakernel::qkv_simt::qkv_24w_default, true, true, 0,
                        -1, false, true>),
                   "opcode1_simt_24w_pdl_wait_trigger_no_prefetch_"
                   "fused_context1_attention",
                   megakernel::qkv_simt::grid_ctas,
                   megakernel::qkv_simt::qkv_24w_default::threads);
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<
                        megakernel::qkv_simt::qkv_8w_default, true, true, 0,
                        -1, false, true, true>),
                   "opcode1_simt_8w_pdl_wait_trigger_no_prefetch_"
                   "fused_context1_attention_kv_only",
                   megakernel::qkv_simt::kv_only_grid_ctas,
                   megakernel::qkv_simt::qkv_8w_default::threads);
#define BIND_FUSED_QKV_POLICY(policy_value, policy_name)                     \
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<                  \
                        megakernel::qkv_simt::qkv_24w_default, true, true,  \
                        1, -1, false, true, false, policy_value>),          \
                   "opcode1_simt_24w_pdl_wait_trigger_prefetch128_"       \
                   policy_name "_fused_context1_attention",              \
                   megakernel::qkv_simt::grid_ctas,                        \
                   megakernel::qkv_simt::qkv_24w_default::threads);        \
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<                  \
                        megakernel::qkv_simt::qkv_8w_default, true, true,   \
                        1, -1, false, true, true, policy_value>),           \
                   "opcode1_simt_8w_pdl_wait_trigger_prefetch128_"        \
                   policy_name "_fused_context1_attention_kv_only",      \
                   megakernel::qkv_simt::kv_only_grid_ctas,                \
                   megakernel::qkv_simt::qkv_8w_default::threads)
    BIND_FUSED_QKV_POLICY(megakernel::mlp_simt::prefetch_policy_l1, "l1");
    BIND_FUSED_QKV_POLICY(
        megakernel::mlp_simt::prefetch_policy_l2_evict_last,
        "l2_evict_last");
#undef BIND_FUSED_QKV_POLICY
#define BIND_QKV_KV_ONLY(config_name, warps_name)                             \
    BIND_FIXED((megakernel::qkv_simt::qkv_rope_append<                      \
                   megakernel::qkv_simt::config_name, false, true, 0, -1,  \
                   false, true, true>),                                     \
               "opcode1_simt_" warps_name                                 \
               "w_pdl_trigger_fused_context1_attention_kv_only",          \
               megakernel::qkv_simt::kv_only_grid_ctas,                    \
               megakernel::qkv_simt::config_name::threads);                \
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<                  \
                        megakernel::qkv_simt::config_name, true, true, 1,   \
                        -1, false, true, true>),                            \
                   "opcode1_simt_" warps_name                             \
                   "w_pdl_wait_trigger_prefetch128_fused_context1_"       \
                   "attention_kv_only",                                   \
                   megakernel::qkv_simt::kv_only_grid_ctas,                \
                   megakernel::qkv_simt::config_name::threads)
    BIND_QKV_KV_ONLY(qkv_4w_default, "4");
    BIND_QKV_KV_ONLY(qkv_12w_pair, "12");
    BIND_QKV_KV_ONLY(qkv_16w_pair, "16");
    BIND_QKV_KV_ONLY(qkv_24w_default, "24");
#undef BIND_QKV_KV_ONLY
#define BIND_QKV_WAIT_TRIGGER_128_PAIR(pair_value, pair_name)                \
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<                  \
                        megakernel::qkv_simt::qkv_24w_default, true, true,  \
                        1, pair_value>),                                     \
                   "opcode1_simt_24w_pdl_wait_trigger_prefetch128_"        \
                   pair_name,                                               \
                   megakernel::qkv_simt::grid_ctas,                        \
                   megakernel::qkv_simt::qkv_24w_default::threads)
    BIND_QKV_WAIT_TRIGGER_128_PAIR(0, "mainloop");
    BIND_QKV_WAIT_TRIGGER_128_PAIR(512, "k50");
    BIND_QKV_WAIT_TRIGGER_128_PAIR(768, "k75");
    BIND_QKV_WAIT_TRIGGER_128_PAIR(896, "k875");
    BIND_QKV_WAIT_TRIGGER_128_PAIR(960, "k9375");
    BIND_QKV_WAIT_TRIGGER_128_PAIR(1024, "epilogue");
#undef BIND_QKV_WAIT_TRIGGER_128_PAIR
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<
                        megakernel::qkv_simt::qkv_24w_default, true, true, 1,
                        -1, true>),
                   "opcode1_simt_24w_pdl_wait_trigger_prefetch128_vready",
                   megakernel::qkv_simt::grid_ctas,
                   megakernel::qkv_simt::qkv_24w_default::threads);
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<
                        megakernel::qkv_simt::qkv_24w_default, true, true, 2>),
                   "opcode1_simt_24w_pdl_wait_trigger_prefetch256",
                   megakernel::qkv_simt::grid_ctas,
                   megakernel::qkv_simt::qkv_24w_default::threads);
#define BIND_QKV_WAIT_TRIGGER_256_PAIR(pair_value, pair_name)                \
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<                  \
                        megakernel::qkv_simt::qkv_24w_default, true, true,  \
                        2, pair_value>),                                     \
                   "opcode1_simt_24w_pdl_wait_trigger_prefetch256_"        \
                   pair_name,                                               \
                   megakernel::qkv_simt::grid_ctas,                        \
                   megakernel::qkv_simt::qkv_24w_default::threads)
    BIND_QKV_WAIT_TRIGGER_256_PAIR(0, "mainloop");
    BIND_QKV_WAIT_TRIGGER_256_PAIR(512, "k50");
    BIND_QKV_WAIT_TRIGGER_256_PAIR(768, "k75");
    BIND_QKV_WAIT_TRIGGER_256_PAIR(896, "k875");
    BIND_QKV_WAIT_TRIGGER_256_PAIR(960, "k9375");
    BIND_QKV_WAIT_TRIGGER_256_PAIR(1024, "epilogue");
#undef BIND_QKV_WAIT_TRIGGER_256_PAIR
#define BIND_QKV_WAIT_TRIGGER_256_TAIL(tail_value, tail_name)              \
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<                \
                        megakernel::qkv_simt::qkv_24w_default, true, true,\
                        2, -1, false, false, false,                       \
                        megakernel::mlp_simt::prefetch_policy_l2, false,  \
                        tail_value>),                                     \
                   "opcode1_simt_24w_pdl_wait_trigger_prefetch256_"      \
                   tail_name,                                            \
                   megakernel::qkv_simt::grid_ctas,                       \
                   megakernel::qkv_simt::qkv_24w_default::threads)
    BIND_QKV_WAIT_TRIGGER_256_TAIL(1, "tail1");
    BIND_QKV_WAIT_TRIGGER_256_TAIL(2, "tail2");
    BIND_QKV_WAIT_TRIGGER_256_TAIL(4, "tail4");
#undef BIND_QKV_WAIT_TRIGGER_256_TAIL
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<
                        megakernel::qkv_simt::qkv_24w_default, true, true, 4>),
                   "opcode1_simt_24w_pdl_wait_trigger_prefetch512",
                   megakernel::qkv_simt::grid_ctas,
                   megakernel::qkv_simt::qkv_24w_default::threads);
    BIND_FIXED_PDL((megakernel::qkv_simt::qkv_rope_append<
                        megakernel::qkv_simt::qkv_24w_default, true, true, 8>),
                   "opcode1_simt_24w_pdl_wait_trigger_prefetch1024",
                   megakernel::qkv_simt::grid_ctas,
                   megakernel::qkv_simt::qkv_24w_default::threads);
    BIND_FIXED((megakernel::qkv_simt::qkv_rope_append<
                   megakernel::qkv_simt::qkv_24w_stream>),
               "opcode1_simt_24w_stream", megakernel::qkv_simt::grid_ctas,
               megakernel::qkv_simt::qkv_24w_stream::threads);

    BIND_FIXED((megakernel::oproj_simt::oproj_residual<
                   megakernel::oproj_simt::oproj_128cta_16w>),
               "opcode4_simt_128cta_16w",
               megakernel::oproj_simt::oproj_128cta_16w::ctas,
               megakernel::oproj_simt::oproj_128cta_16w::threads);
    BIND_FIXED((megakernel::oproj_simt::oproj_residual<
                   megakernel::oproj_simt::oproj_128cta_8w2r>),
               "opcode4_simt_128cta_8w2r",
               megakernel::oproj_simt::oproj_128cta_8w2r::ctas,
               megakernel::oproj_simt::oproj_128cta_8w2r::threads);
    BIND_FIXED((megakernel::oproj_simt::oproj_residual<
                   megakernel::oproj_simt::oproj_132cta_16w>),
               "opcode4_simt_132cta_16w",
               megakernel::oproj_simt::oproj_132cta_16w::ctas,
               megakernel::oproj_simt::oproj_132cta_16w::threads);
    BIND_FIXED((megakernel::oproj_simt::oproj_residual<
                   megakernel::oproj_simt::oproj_132cta_16w_sync>),
               "opcode4_simt_132cta_16w_sync",
               megakernel::oproj_simt::oproj_132cta_16w_sync::ctas,
               megakernel::oproj_simt::oproj_132cta_16w_sync::threads);
    BIND_FIXED((megakernel::oproj_simt::oproj_residual<
                   megakernel::oproj_simt::oproj_132cta_16w_cp_async>),
               "opcode4_simt_132cta_16w_cp_async",
               megakernel::oproj_simt::oproj_132cta_16w_cp_async::ctas,
               megakernel::oproj_simt::oproj_132cta_16w_cp_async::threads);
    BIND_FIXED((megakernel::oproj_simt::oproj_residual_v4<
                   megakernel::oproj_simt::oproj_128cta_8w2r, 0>),
               "opcode4_simt_128cta_8w2r_v4",
               megakernel::oproj_simt::oproj_128cta_8w2r::ctas,
               megakernel::oproj_simt::oproj_128cta_8w2r::threads);
    BIND_FIXED((megakernel::oproj_simt::oproj_residual_v4<
                   megakernel::oproj_simt::oproj_128cta_8w2r, -3>),
               "opcode4_simt_128cta_8w2r_v4_na",
               megakernel::oproj_simt::oproj_128cta_8w2r::ctas,
               megakernel::oproj_simt::oproj_128cta_8w2r::threads);
    BIND_FIXED((megakernel::oproj_simt::oproj_residual_v4<
                   megakernel::oproj_simt::oproj_128cta_8w2r, -5>),
               "opcode4_simt_128cta_8w2r_v4_na_l2_256b",
               megakernel::oproj_simt::oproj_128cta_8w2r::ctas,
               megakernel::oproj_simt::oproj_128cta_8w2r::threads);
    BIND_FIXED((megakernel::oproj_simt::oproj_residual_v4<
                   megakernel::oproj_simt::oproj_128cta_16w, 0>),
               "opcode4_simt_128cta_16w_v4",
               megakernel::oproj_simt::oproj_128cta_16w::ctas,
               megakernel::oproj_simt::oproj_128cta_16w::threads);
    BIND_FIXED((megakernel::oproj_simt::oproj_residual_v4<
                   megakernel::oproj_simt::oproj_128cta_16w, -3>),
               "opcode4_simt_128cta_16w_v4_na",
               megakernel::oproj_simt::oproj_128cta_16w::ctas,
               megakernel::oproj_simt::oproj_128cta_16w::threads);
    BIND_FIXED((megakernel::oproj_simt::oproj_residual_v4<
                   megakernel::oproj_simt::oproj_128cta_16w, -3, true>),
               "opcode4_simt_128cta_16w_v4_na_pdl_trigger",
               megakernel::oproj_simt::oproj_128cta_16w::ctas,
               megakernel::oproj_simt::oproj_128cta_16w::threads);
#define BIND_OP4_TRIGGER_QUAD(quad_value, quad_name)                          \
    BIND_FIXED((megakernel::oproj_simt::oproj_residual_v4<                   \
                   megakernel::oproj_simt::oproj_128cta_16w, -3, true,      \
                   false, quad_value>),                                      \
               "opcode4_simt_128cta_16w_v4_na_pdl_trigger_" quad_name,       \
               megakernel::oproj_simt::oproj_128cta_16w::ctas,              \
               megakernel::oproj_simt::oproj_128cta_16w::threads)
    BIND_OP4_TRIGGER_QUAD(256, "k50");
    BIND_OP4_TRIGGER_QUAD(384, "k75");
    BIND_OP4_TRIGGER_QUAD(448, "k875");
    BIND_OP4_TRIGGER_QUAD(480, "k9375");
    BIND_OP4_TRIGGER_QUAD(512, "epilogue");
#undef BIND_OP4_TRIGGER_QUAD
    BIND_FIXED_PDL((megakernel::oproj_simt::oproj_residual_v4<
                        megakernel::oproj_simt::oproj_128cta_16w, -3, true,
                        true>),
                   "opcode4_simt_128cta_16w_v4_na_pdl_wait_trigger",
                   megakernel::oproj_simt::oproj_128cta_16w::ctas,
                   megakernel::oproj_simt::oproj_128cta_16w::threads);
#define BIND_OP4_WAIT_TRIGGER_QUAD(quad_value, quad_name)                    \
    BIND_FIXED_PDL((megakernel::oproj_simt::oproj_residual_v4<              \
                        megakernel::oproj_simt::oproj_128cta_16w, -3, true, \
                        true, quad_value>),                                  \
                   "opcode4_simt_128cta_16w_v4_na_pdl_wait_trigger_"       \
                   quad_name,                                               \
                   megakernel::oproj_simt::oproj_128cta_16w::ctas,          \
                   megakernel::oproj_simt::oproj_128cta_16w::threads)
    BIND_OP4_WAIT_TRIGGER_QUAD(256, "k50");
    BIND_OP4_WAIT_TRIGGER_QUAD(384, "k75");
    BIND_OP4_WAIT_TRIGGER_QUAD(448, "k875");
    BIND_OP4_WAIT_TRIGGER_QUAD(480, "k9375");
#undef BIND_OP4_WAIT_TRIGGER_QUAD
#define BIND_OP4_WAIT_PREFETCH(lines_value, bytes_name)                     \
    BIND_FIXED_PDL((megakernel::oproj_simt::oproj_residual_v4<             \
                        megakernel::oproj_simt::oproj_128cta_16w, -3,      \
                        false, true, -1, lines_value>),                     \
                   "opcode4_simt_128cta_16w_v4_na_pdl_wait_prefetch"      \
                   bytes_name,                                             \
                   megakernel::oproj_simt::oproj_128cta_16w::ctas,        \
                   megakernel::oproj_simt::oproj_128cta_16w::threads)
#define BIND_OP4_WAIT_PREFETCH_PLAIN(lines_value, bytes_name)               \
    BIND_FIXED((megakernel::oproj_simt::oproj_residual_v4<                 \
                   megakernel::oproj_simt::oproj_128cta_16w, -3, false,    \
                   true, -1, lines_value>),                                \
               "opcode4_simt_128cta_16w_v4_na_pdl_wait_prefetch"         \
               bytes_name "_plain_launch",                               \
               megakernel::oproj_simt::oproj_128cta_16w::ctas,            \
               megakernel::oproj_simt::oproj_128cta_16w::threads)
    BIND_FIXED_PDL((megakernel::oproj_simt::oproj_residual_v4<
                        megakernel::oproj_simt::oproj_128cta_16w, -3,
                        false, true>),
                   "opcode4_simt_128cta_16w_v4_na_pdl_wait",
                   megakernel::oproj_simt::oproj_128cta_16w::ctas,
                   megakernel::oproj_simt::oproj_128cta_16w::threads);
    BIND_OP4_WAIT_PREFETCH(1, "128b");
    BIND_OP4_WAIT_PREFETCH(2, "256b");
    BIND_OP4_WAIT_PREFETCH(4, "512b");
    BIND_OP4_WAIT_PREFETCH(8, "1k");
    BIND_OP4_WAIT_PREFETCH(16, "2k");
    BIND_OP4_WAIT_PREFETCH(32, "4k");
#undef BIND_OP4_WAIT_PREFETCH
    BIND_FIXED_PDL((megakernel::oproj_simt::oproj_residual_v4<
                        megakernel::oproj_simt::oproj_128cta_16w, -3, false,
                        true, -1, 1,
                        megakernel::mlp_simt::prefetch_policy_l1>),
                   "opcode4_simt_128cta_16w_v4_na_pdl_wait_prefetch128b_l1",
                   megakernel::oproj_simt::oproj_128cta_16w::ctas,
                   megakernel::oproj_simt::oproj_128cta_16w::threads);
    BIND_FIXED_PDL((megakernel::oproj_simt::oproj_residual_v4<
                        megakernel::oproj_simt::oproj_128cta_16w, -3, false,
                        true, -1, 1,
                        megakernel::mlp_simt::prefetch_policy_l2_evict_last>),
                   "opcode4_simt_128cta_16w_v4_na_pdl_wait_prefetch128b_"
                   "l2_evict_last",
                   megakernel::oproj_simt::oproj_128cta_16w::ctas,
                   megakernel::oproj_simt::oproj_128cta_16w::threads);
    BIND_OP4_WAIT_PREFETCH_PLAIN(1, "128b");
    BIND_OP4_WAIT_PREFETCH_PLAIN(2, "256b");
    BIND_OP4_WAIT_PREFETCH_PLAIN(4, "512b");
    BIND_OP4_WAIT_PREFETCH_PLAIN(8, "1k");
    BIND_OP4_WAIT_PREFETCH_PLAIN(16, "2k");
    BIND_OP4_WAIT_PREFETCH_PLAIN(32, "4k");
#undef BIND_OP4_WAIT_PREFETCH_PLAIN
    BIND_FIXED_PDL((megakernel::oproj_simt::oproj_residual_v4<
                        megakernel::oproj_simt::oproj_128cta_16w, -3, true,
                        true, 512, 1>),
                   "opcode4_simt_128cta_16w_v4_na_pdl_wait_prefetch128b_"
                   "trigger_epilogue",
                   megakernel::oproj_simt::oproj_128cta_16w::ctas,
                   megakernel::oproj_simt::oproj_128cta_16w::threads);
    BIND_FIXED_PDL((megakernel::oproj_simt::oproj_residual_v4<
                        megakernel::oproj_simt::oproj_128cta_16w, -3, true,
                        true, -1, 1>),
                   "opcode4_simt_128cta_16w_v4_na_pdl_wait_prefetch128b_"
                   "trigger_entry",
                   megakernel::oproj_simt::oproj_128cta_16w::ctas,
                   megakernel::oproj_simt::oproj_128cta_16w::threads);
    BIND_FIXED_PDL((megakernel::oproj_simt::oproj_residual_v4<
                        megakernel::oproj_simt::oproj_128cta_16w, -3, true,
                        true, 512, 0>),
                   "opcode4_simt_128cta_16w_v4_na_pdl_wait_trigger_"
                   "epilogue",
                   megakernel::oproj_simt::oproj_128cta_16w::ctas,
                   megakernel::oproj_simt::oproj_128cta_16w::threads);

#define BIND_OP5_WAIT_PREFETCH(lines_value, bytes_name)                      \
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_pairwarp<               \
                        megakernel::mlp_simt::upgate_32w2r_132, -5, true,    \
                        true, true, -1, lines_value>),                        \
                   "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"     \
                   "wait_trigger_prefetch" bytes_name,                      \
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,             \
                   megakernel::mlp_simt::upgate_32w2r_132::threads)
    BIND_OP5_WAIT_PREFETCH(1, "128b");
    BIND_OP5_WAIT_PREFETCH(2, "256b");
    BIND_OP5_WAIT_PREFETCH(4, "512b");
    BIND_OP5_WAIT_PREFETCH(8, "1k");
#undef BIND_OP5_WAIT_PREFETCH
#define BIND_OP5_WAIT_PREFETCH128_TRIGGER_PAIR(pair_value, pair_name)        \
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_pairwarp<               \
                        megakernel::mlp_simt::upgate_32w2r_132, -5, true,    \
                        true, true, pair_value, 1>),                          \
                   "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"    \
                   "wait_trigger_" pair_name "_prefetch128b",             \
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,             \
                   megakernel::mlp_simt::upgate_32w2r_132::threads)
    BIND_OP5_WAIT_PREFETCH128_TRIGGER_PAIR(512, "k50");
    BIND_OP5_WAIT_PREFETCH128_TRIGGER_PAIR(768, "k75");
    BIND_OP5_WAIT_PREFETCH128_TRIGGER_PAIR(896, "k875");
    BIND_OP5_WAIT_PREFETCH128_TRIGGER_PAIR(960, "k9375");
    BIND_OP5_WAIT_PREFETCH128_TRIGGER_PAIR(1024, "epilogue");
#undef BIND_OP5_WAIT_PREFETCH128_TRIGGER_PAIR
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_pairwarp<
                        megakernel::mlp_simt::upgate_32w2r_132, -5, true,
                        true, true, -1, 1, 3, 0, false,
                        megakernel::mlp_simt::prefetch_policy_l1>),
                   "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"
                   "wait_trigger_prefetch128b_l1",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_pairwarp<
                        megakernel::mlp_simt::upgate_32w2r_132, -5, true,
                        true, true, -1, 1, 3, 0, false,
                        megakernel::mlp_simt::prefetch_policy_l2_evict_last>),
                   "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"
                   "wait_trigger_prefetch128b_l2_evict_last",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_pairwarp<
                        megakernel::mlp_simt::upgate_32w2r_132, -5, true,
                        true, true, -1, 1, 3, 0, false,
                        megakernel::mlp_simt::prefetch_policy_l2, true>),
                   "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_v4_pdl_"
                   "wait_trigger_prefetch128b",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
#define BIND_OP5_WAIT_ONLY_PREFETCH(lines_value, bytes_name)                 \
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_pairwarp<              \
                        megakernel::mlp_simt::upgate_32w2r_132, -5, true,   \
                        false, true, -1, lines_value>),                      \
                   "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"  \
                   "wait_prefetch" bytes_name,                             \
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,            \
                   megakernel::mlp_simt::upgate_32w2r_132::threads)
#define BIND_OP5_WAIT_ONLY_PREFETCH_PLAIN(lines_value, bytes_name)           \
    BIND_FIXED((megakernel::mlp_simt::upgate_132_pairwarp<                  \
                   megakernel::mlp_simt::upgate_32w2r_132, -5, true,       \
                   false, true, -1, lines_value>),                          \
               "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"     \
               "wait_prefetch" bytes_name "_plain_launch",              \
               megakernel::mlp_simt::upgate_32w2r_132::ctas,               \
               megakernel::mlp_simt::upgate_32w2r_132::threads)
    BIND_OP5_WAIT_ONLY_PREFETCH(0, "0b");
    BIND_OP5_WAIT_ONLY_PREFETCH(1, "128b");
    BIND_OP5_WAIT_ONLY_PREFETCH(2, "256b");
    BIND_OP5_WAIT_ONLY_PREFETCH(4, "512b");
    BIND_OP5_WAIT_ONLY_PREFETCH(8, "1k");
#undef BIND_OP5_WAIT_ONLY_PREFETCH
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_pairwarp<
                        megakernel::mlp_simt::upgate_32w2r_132, -5, true,
                        false, true, -1, 2, 3, 0, false,
                        megakernel::mlp_simt::prefetch_policy_l1>),
                   "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"
                   "wait_prefetch256b_l1",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_pairwarp<
                        megakernel::mlp_simt::upgate_32w2r_132, -5, true,
                        false, true, -1, 2, 3, 0, false,
                        megakernel::mlp_simt::prefetch_policy_l2_evict_last>),
                   "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"
                   "wait_prefetch256b_l2_evict_last",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_OP5_WAIT_ONLY_PREFETCH_PLAIN(0, "0b");
    BIND_OP5_WAIT_ONLY_PREFETCH_PLAIN(1, "128b");
    BIND_OP5_WAIT_ONLY_PREFETCH_PLAIN(2, "256b");
    BIND_OP5_WAIT_ONLY_PREFETCH_PLAIN(4, "512b");
    BIND_OP5_WAIT_ONLY_PREFETCH_PLAIN(8, "1k");
#undef BIND_OP5_WAIT_ONLY_PREFETCH_PLAIN
// Strong warming controls: each lane-selected instruction is a real demand
// load with a 256-byte L2 prefetch hint.  Results die before the PDL wait, so
// these do not consume long-lived registers or add a SMEM weight pass.
#define BIND_OP5_WAIT_DEMAND(lines_value, bytes_name)                        \
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_pairwarp<              \
                        megakernel::mlp_simt::upgate_32w2r_132, -5, true,   \
                        true, true, -1, lines_value, 3, 256>),               \
                   "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"   \
                   "wait_trigger_demand" bytes_name,                       \
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,            \
                   megakernel::mlp_simt::upgate_32w2r_132::threads)
    BIND_OP5_WAIT_DEMAND(1, "256b");
    BIND_OP5_WAIT_DEMAND(2, "512b");
    BIND_OP5_WAIT_DEMAND(4, "1k");
    BIND_OP5_WAIT_DEMAND(8, "2k");
    BIND_OP5_WAIT_DEMAND(16, "4k");
#undef BIND_OP5_WAIT_DEMAND
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_pairwarp<
                        megakernel::mlp_simt::upgate_32w2r_132, -5, true,
                        true, true, -1, 1, 1>),
                   "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"
                   "wait_trigger_prefetch128b_up_only",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_pairwarp<
                        megakernel::mlp_simt::upgate_32w2r_132, -5, true,
                        true, true, -1, 1, 2>),
                   "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"
                   "wait_trigger_prefetch128b_gate_only",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_pairwarp<
                        megakernel::mlp_simt::upgate_32w2r_132, -5, true,
                        true, true, -1, 1, 3, 0, true>),
                   "opcode5_simt_32w2r_i_132cta_allwarp_na_l2_256b_pdl_"
                   "wait_trigger_prefetch128b_split_ready",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_split4_pipeline<
                        megakernel::mlp_simt::upgate_32w2r_132>),
                   "opcode5_simt_32w2r_i_132cta_split4_pipeline_pdl_"
                   "wait_trigger_prefetch128b",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_split2_pipeline<
                        megakernel::mlp_simt::upgate_32w2r_132, false>),
                   "opcode5_simt_32w1r_i_132cta_split2_pipeline_pdl_"
                   "wait_trigger_prefetch128b",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_split2_pipeline<
                        megakernel::mlp_simt::upgate_32w2r_132, true>),
                   "opcode5_simt_16w2r_i_132cta_split2_pipeline_pdl_"
                   "wait_trigger_prefetch128b",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_register_prefix_pdl<
                        megakernel::mlp_simt::upgate_32w2r_132, 1>),
                   "opcode5_simt_32w2r_i_132cta_regprefix64_pdl_wait_"
                   "trigger",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_register_prefix_pdl<
                        megakernel::mlp_simt::upgate_32w2r_132, 2>),
                   "opcode5_simt_32w2r_i_132cta_regprefix128_pdl_wait_"
                   "trigger",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_register_prefix_pdl<
                        megakernel::mlp_simt::upgate_32w2r_132, 1, false>),
                   "opcode5_simt_32w2r_i_132cta_regprefix64_pdl_wait",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED_PDL((megakernel::mlp_simt::upgate_132_register_prefix_pdl<
                        megakernel::mlp_simt::upgate_32w2r_132, 2, false>),
                   "opcode5_simt_32w2r_i_132cta_regprefix128_pdl_wait",
                   megakernel::mlp_simt::upgate_32w2r_132::ctas,
                   megakernel::mlp_simt::upgate_32w2r_132::threads);
    BIND_FIXED((megakernel::oproj_simt::oproj_residual_v4<
                   megakernel::oproj_simt::oproj_128cta_16w, -5>),
               "opcode4_simt_128cta_16w_v4_na_l2_256b",
               megakernel::oproj_simt::oproj_128cta_16w::ctas,
               megakernel::oproj_simt::oproj_128cta_16w::threads);

    BIND_FIXED((megakernel::attention_simt::context1_attention<
                   megakernel::attention_simt::attention_1cta_32w>),
               "opcode2_simt_1cta_32w",
               megakernel::attention_simt::attention_1cta_32w::ctas,
               megakernel::attention_simt::attention_1cta_32w::threads);
    BIND_FIXED((megakernel::attention_simt::context1_attention<
                   megakernel::attention_simt::attention_4cta_8w>),
               "opcode2_simt_4cta_8w",
               megakernel::attention_simt::attention_4cta_8w::ctas,
               megakernel::attention_simt::attention_4cta_8w::threads);
    BIND_FIXED((megakernel::attention_simt::context1_attention<
                   megakernel::attention_simt::attention_8cta_4w>),
               "opcode2_simt_8cta_4w",
               megakernel::attention_simt::attention_8cta_4w::ctas,
               megakernel::attention_simt::attention_8cta_4w::threads);
    BIND_FIXED_PDL((megakernel::attention_simt::context1_attention_bypass<
                        megakernel::attention_simt::attention_8cta_4w, true>),
                   "opcode2_simt_8cta_4w_pdl_wait_bypass",
                   megakernel::attention_simt::attention_8cta_4w::ctas,
                   megakernel::attention_simt::attention_8cta_4w::threads);
    BIND_FIXED_PDL((megakernel::attention_simt::context1_attention_bypass<
                        megakernel::attention_simt::attention_bypass_1cta_1w,
                        true>),
                   "opcode2_simt_1cta_1w_pdl_wait_bypass",
                   megakernel::attention_simt::attention_bypass_1cta_1w::ctas,
                   megakernel::attention_simt::attention_bypass_1cta_1w::threads);
    BIND_FIXED((megakernel::attention_simt::context1_attention<
                   megakernel::attention_simt::attention_8cta_4w, false,
                   true>),
               "opcode2_simt_8cta_4w_pdl_trigger",
               megakernel::attention_simt::attention_8cta_4w::ctas,
               megakernel::attention_simt::attention_8cta_4w::threads);
    BIND_FIXED((megakernel::attention_simt::context1_attention<
                   megakernel::attention_simt::attention_8cta_4w, false,
                   true, 0>),
               "opcode2_simt_8cta_4w_pdl_trigger_epilogue",
               megakernel::attention_simt::attention_8cta_4w::ctas,
               megakernel::attention_simt::attention_8cta_4w::threads);
    BIND_FIXED_PDL((megakernel::attention_simt::context1_attention<
                        megakernel::attention_simt::attention_8cta_4w, true>),
                   "opcode2_simt_8cta_4w_pdl_wait",
                   megakernel::attention_simt::attention_8cta_4w::ctas,
                   megakernel::attention_simt::attention_8cta_4w::threads);
    BIND_FIXED((megakernel::attention_simt::context1_attention<
                    megakernel::attention_simt::attention_8cta_4w, true>),
               "opcode2_simt_8cta_4w_pdl_wait_plain_launch",
               megakernel::attention_simt::attention_8cta_4w::ctas,
               megakernel::attention_simt::attention_8cta_4w::threads);
    BIND_FIXED_PDL((megakernel::attention_simt::context1_attention<
                        megakernel::attention_simt::attention_8cta_4w, false,
                        false, -1, true>),
                   "opcode2_simt_8cta_4w_vready_wait",
                   megakernel::attention_simt::attention_8cta_4w::ctas,
                   megakernel::attention_simt::attention_8cta_4w::threads);
    BIND_FIXED_PDL((megakernel::attention_simt::context1_attention<
                        megakernel::attention_simt::attention_8cta_4w, true,
                        true>),
                   "opcode2_simt_8cta_4w_pdl_wait_trigger",
                   megakernel::attention_simt::attention_8cta_4w::ctas,
                   megakernel::attention_simt::attention_8cta_4w::threads);
    BIND_FIXED_PDL((megakernel::attention_simt::context1_attention<
                        megakernel::attention_simt::attention_8cta_4w, true,
                        true, 0>),
                   "opcode2_simt_8cta_4w_pdl_wait_trigger_epilogue",
                   megakernel::attention_simt::attention_8cta_4w::ctas,
                   megakernel::attention_simt::attention_8cta_4w::threads);
    BIND_FIXED((megakernel::attention_simt::context1_attention<
                   megakernel::attention_simt::attention_32cta_1w>),
               "opcode2_simt_32cta_1w",
               megakernel::attention_simt::attention_32cta_1w::ctas,
               megakernel::attention_simt::attention_32cta_1w::threads);

    BIND_FIXED((megakernel::attention_simt::short_context_attention<>),
               "opcode2_simt_short_8cta_4w_cp_async_smem",
               megakernel::attention_simt::short_context_8cta_4w::ctas,
               megakernel::attention_simt::short_context_8cta_4w::threads);
    BIND_FIXED((megakernel::attention_simt::short_context_attention<
                   false, false, false, false>),
               "opcode2_simt_short_8cta_4w_ldg_smem",
               megakernel::attention_simt::short_context_8cta_4w::ctas,
               megakernel::attention_simt::short_context_8cta_4w::threads);
    BIND_FIXED_PDL(
        (megakernel::attention_simt::short_context_attention<true, false,
                                                              true>),
        "opcode2_simt_short_8cta_4w_pdl_wait_prefetch_kv",
        megakernel::attention_simt::short_context_8cta_4w::ctas,
        megakernel::attention_simt::short_context_8cta_4w::threads);
    BIND_FIXED((megakernel::attention_simt::short_context_attention<false,
                                                                    true>),
               "opcode2_simt_short_8cta_4w_pdl_trigger",
               megakernel::attention_simt::short_context_8cta_4w::ctas,
               megakernel::attention_simt::short_context_8cta_4w::threads);
    BIND_FIXED_PDL(
        (megakernel::attention_simt::short_context_attention<true, true,
                                                              true>),
        "opcode2_simt_short_8cta_4w_pdl_wait_trigger_prefetch_kv",
        megakernel::attention_simt::short_context_8cta_4w::ctas,
        megakernel::attention_simt::short_context_8cta_4w::threads);
    BIND_FIXED(
        (megakernel::attention_simt::short_context_attention_direct_gqa<>),
        "opcode2_simt_direct_gqa_8cta_1w",
        megakernel::attention_simt::short_context_8cta_4w::ctas, 32);
    BIND_FIXED_PDL(
        (megakernel::attention_simt::short_context_attention_direct_gqa<
            true, false>),
        "opcode2_simt_direct_gqa_8cta_1w_pdl_wait",
        megakernel::attention_simt::short_context_8cta_4w::ctas, 32);
    BIND_FIXED(
        (megakernel::attention_simt::short_context_attention_direct_gqa<
            false, true>),
        "opcode2_simt_direct_gqa_8cta_1w_pdl_trigger",
        megakernel::attention_simt::short_context_8cta_4w::ctas, 32);
    BIND_FIXED_PDL(
        (megakernel::attention_simt::short_context_attention_direct_gqa<
            true, true>),
        "opcode2_simt_direct_gqa_8cta_1w_pdl_wait_trigger",
        megakernel::attention_simt::short_context_8cta_4w::ctas, 32);
    BIND_FIXED_PDL(
        (megakernel::attention_simt::short_context_attention_direct_gqa<
            true, true, 2>),
        "opcode2_simt_direct_gqa_8cta_2w_pdl_wait_trigger",
        megakernel::attention_simt::short_context_8cta_4w::ctas, 64);
    BIND_FIXED_PDL(
        (megakernel::attention_simt::short_context_attention_direct_gqa<
            true, true, 4>),
        "opcode2_simt_direct_gqa_8cta_4w_pdl_wait_trigger",
        megakernel::attention_simt::short_context_8cta_4w::ctas, 128);
#define BIND_DIRECT_GQA_REG_PREFETCH(warps_value, tokens_value)              \
    BIND_FIXED_PDL(                                                          \
        (megakernel::attention_simt::short_context_attention_direct_gqa<    \
            true, true, warps_value, tokens_value>),                        \
        "opcode2_simt_direct_gqa_8cta_" #warps_value                       \
        "w_regprefetch" #tokens_value "_pdl_wait_trigger",               \
        megakernel::attention_simt::short_context_8cta_4w::ctas,            \
        32 * warps_value)
    BIND_DIRECT_GQA_REG_PREFETCH(1, 8);
    BIND_DIRECT_GQA_REG_PREFETCH(1, 16);
    BIND_DIRECT_GQA_REG_PREFETCH(1, 32);
    BIND_DIRECT_GQA_REG_PREFETCH(2, 8);
    BIND_DIRECT_GQA_REG_PREFETCH(2, 16);
    BIND_DIRECT_GQA_REG_PREFETCH(2, 32);
    BIND_DIRECT_GQA_REG_PREFETCH(4, 8);
    BIND_DIRECT_GQA_REG_PREFETCH(4, 16);
    BIND_DIRECT_GQA_REG_PREFETCH(4, 32);
#undef BIND_DIRECT_GQA_REG_PREFETCH
    BIND_FIXED_PDL(
        (megakernel::attention_simt::
             short_context_attention_direct_gqa_block16_reg32<>),
        "opcode2_simt_direct_gqa_8cta_4w_block16_reg32_pdl_wait_trigger",
        megakernel::attention_simt::short_context_8cta_4w::ctas, 128);

    BIND_FIXED_SMEM(
        (megakernel::attention_tk_short::short_attention<>),
        "opcode2_tk_short_8cta_1w",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<true, false>),
        "opcode2_tk_short_8cta_1w_pdl_wait",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes);
    BIND_FIXED_SMEM(
        (megakernel::attention_tk_short::short_attention<false, true>),
        "opcode2_tk_short_8cta_1w_pdl_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<true, true>),
        "opcode2_tk_short_8cta_1w_pdl_wait_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<true, true, true>),
        "opcode2_tk_short_8cta_1w_pdl_late_current_kv_wait_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<
            true, true, true, 3, false, true>),
        "opcode2_tk_short_cpasync_8cta_1w_s3_pdl_late_current_kv_wait_"
        "trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes_for<3>);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<
            true, true, true, 10, false, true>),
        "opcode2_tk_short_cpasync_8cta_1w_s10_pdl_late_current_kv_wait_"
        "trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes_for<10>);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<true, true, false, 4>),
        "opcode2_tk_short_8cta_1w_s4_pdl_wait_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes_for<4>);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<true, true, false, 5>),
        "opcode2_tk_short_8cta_1w_s5_pdl_wait_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes_for<5>);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<true, true, false, 10>),
        "opcode2_tk_short_8cta_1w_s10_pdl_wait_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes_for<10>);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<true, true, false, 3,
                                                         true>),
        "opcode2_tk_short_8cta_1w_s3_hist_pdl_wait_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes_for<3>);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<true, true, false, 5,
                                                         true>),
        "opcode2_tk_short_8cta_1w_s5_hist_pdl_wait_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes_for<5>);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<true, true, false, 10,
                                                         true>),
        "opcode2_tk_short_8cta_1w_s10_hist_pdl_wait_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes_for<10>);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<true, true, false, 3,
                                                         false, true>),
        "opcode2_tk_short_cpasync_8cta_1w_s3_pdl_wait_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes_for<3>);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<true, true, false, 3,
                                                         true, true>),
        "opcode2_tk_short_cpasync_8cta_1w_s3_hist_pdl_wait_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes_for<3>);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<true, true, false, 10,
                                                         true, true>),
        "opcode2_tk_short_cpasync_8cta_1w_s10_hist_pdl_wait_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes_for<10>);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<true, true, false, 10,
                                                         false, true>),
        "opcode2_tk_short_cpasync_8cta_1w_s10_pdl_wait_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes_for<10>);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<true, true, false, 10,
                                                         true, true, 6>),
        "opcode2_tk_short_cpasync_8cta_1w_s10_hybrid6_pdl_wait_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes_for<10>);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<
            true, true, false, 3, false, false, 0, true>),
        "opcode2_tk_short_reg1_8cta_1w_pdl_wait_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes_for<3>);
    BIND_FIXED_SMEM_PDL(
        (megakernel::attention_tk_short::short_attention<
            true, true, false, 3, false, true, 0, true>),
        "opcode2_tk_short_cpasync_reg1_8cta_1w_pdl_wait_trigger",
        megakernel::attention_tk_short::grid_ctas,
        megakernel::attention_tk_short::threads,
        megakernel::attention_tk_short::dynamic_smem_bytes_for<3>);

    BIND_FIXED((megakernel::attention_reduction_simt::two_partial_reduction<
                   megakernel::attention_reduction_simt::reduction_1cta_32w>),
               "opcode3_simt_1cta_32w",
               megakernel::attention_reduction_simt::reduction_1cta_32w::ctas,
               megakernel::attention_reduction_simt::reduction_1cta_32w::threads);
    BIND_FIXED((megakernel::attention_reduction_simt::two_partial_reduction<
                   megakernel::attention_reduction_simt::reduction_4cta_8w>),
               "opcode3_simt_4cta_8w",
               megakernel::attention_reduction_simt::reduction_4cta_8w::ctas,
               megakernel::attention_reduction_simt::reduction_4cta_8w::threads);
    BIND_FIXED((megakernel::attention_reduction_simt::two_partial_reduction<
                   megakernel::attention_reduction_simt::reduction_8cta_4w>),
               "opcode3_simt_8cta_4w",
               megakernel::attention_reduction_simt::reduction_8cta_4w::ctas,
               megakernel::attention_reduction_simt::reduction_8cta_4w::threads);
    BIND_FIXED((megakernel::attention_reduction_simt::two_partial_reduction<
                   megakernel::attention_reduction_simt::reduction_32cta_1w>),
               "opcode3_simt_32cta_1w",
               megakernel::attention_reduction_simt::reduction_32cta_1w::ctas,
               megakernel::attention_reduction_simt::reduction_32cta_1w::threads);

    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head<
                   megakernel::lm_head_simt::lm_1r_default>),
               "opcode7_simt_1r_default", megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head<
                   megakernel::lm_head_simt::lm_2r_default>),
               "opcode7_simt_2r_default", megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head<
                   megakernel::lm_head_simt::lm_4r_default>),
               "opcode7_simt_4r_default", megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head<
                   megakernel::lm_head_simt::lm_2r_stream>),
               "opcode7_simt_2r_stream", megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head<
                   megakernel::lm_head_simt::lm_4r_stream>),
               "opcode7_simt_4r_stream", megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head<
                   megakernel::lm_head_simt::lm_8r_stream>),
               "opcode7_simt_8r_stream", megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head<
                   megakernel::lm_head_simt::lm_1r_stream_v4>),
               "opcode7_simt_1r_stream_v4",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head<
                   megakernel::lm_head_simt::lm_2r_default_v4>),
               "opcode7_simt_2r_default_v4",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head<
                   megakernel::lm_head_simt::lm_2r_stream_v4>),
               "opcode7_simt_2r_stream_v4",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head<
                   megakernel::lm_head_simt::lm_4r_stream_v4>),
               "opcode7_simt_4r_stream_v4",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head<
                   megakernel::lm_head_simt::lm_2r_stream_v8>),
               "opcode7_simt_2r_stream_v8",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_balanced_tail_v4<-5>),
               "opcode7_simt_2r_stream_v4_balanced_tail",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_balanced_tail_v4<0>),
               "opcode7_simt_2r_default_v4_balanced_tail",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_balanced_tail_v4<-1>),
               "opcode7_simt_2r_cg_v4_balanced_tail",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_balanced_tail_v4<-2>),
               "opcode7_simt_2r_cs_v4_balanced_tail",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_balanced_tail_v4<-3>),
               "opcode7_simt_2r_na_v4_balanced_tail",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_balanced_tail_v4<-4>),
               "opcode7_simt_2r_na128_v4_balanced_tail",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_balanced_tail_v4<-6>),
               "opcode7_simt_2r_nc_v4_balanced_tail",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_balanced_tail_v4<-7>),
               "opcode7_simt_2r_ncna_v4_balanced_tail",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_balanced_tail_v4<-8>),
               "opcode7_simt_2r_ef_v4_balanced_tail",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_balanced_tail_v4<-3,
                                                                        true>),
               "opcode7_simt_2r_na_v4_balanced_tail_fullunroll",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_balanced_tail_v4<
                   -3, true, true>),
               "opcode7_simt_2r_na_v4_balanced_tail_fullunroll_striped",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED_SMEM((megakernel::lm_head_simt::
                         rms_lm_head_cp_async_balanced_tail<256>),
                    "opcode7_simt_cp256_balanced_tail",
                    megakernel::lm_head_simt::grid_ctas,
                    megakernel::lm_head_simt::threads_per_cta,
                    2 * megakernel::lm_head_simt::warps_per_cta * 2 * 256 *
                        sizeof(megakernel::lm_head_simt::bf16));
    BIND_FIXED_SMEM((megakernel::lm_head_simt::
                         rms_lm_head_cp_async_balanced_tail<512>),
                    "opcode7_simt_cp512_balanced_tail",
                    megakernel::lm_head_simt::grid_ctas,
                    megakernel::lm_head_simt::threads_per_cta,
                    2 * megakernel::lm_head_simt::warps_per_cta * 2 * 512 *
                        sizeof(megakernel::lm_head_simt::bf16));
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_24w4r_balanced_tail),
               "opcode7_simt_24w4r_na_v4_fullunroll_balanced_tail",
               megakernel::lm_head_simt::grid_ctas, 24 * 32);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_32w3r_balanced_tail<0>),
               "opcode7_simt_32w3r_na_v4_fullunroll_balanced_tail",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_32w3r_balanced_tail<1>),
               "opcode7_simt_32w3r_na_v4_fullunroll_prefetch128",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_32w3r_balanced_tail<2>),
               "opcode7_simt_32w3r_na_v4_fullunroll_prefetch256",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_32w3r_balanced_tail<4>),
               "opcode7_simt_32w3r_na_v4_fullunroll_prefetch512",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_32w3r_balanced_tail<8>),
               "opcode7_simt_32w3r_na_v4_fullunroll_prefetch1024",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_32w3r_balanced_tail<12>),
               "opcode7_simt_32w3r_na_v4_fullunroll_prefetch1536",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_32w3r_balanced_tail<16>),
               "opcode7_simt_32w3r_na_v4_fullunroll_prefetch2048",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_32w3r_balanced_tail<24>),
               "opcode7_simt_32w3r_na_v4_fullunroll_prefetch3072",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED((megakernel::lm_head_simt::rms_lm_head_32w3r_balanced_tail<32>),
               "opcode7_simt_32w3r_na_v4_fullunroll_prefetch4096",
               megakernel::lm_head_simt::grid_ctas,
               megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED_PDL((megakernel::lm_head_simt::
                        rms_lm_head_32w3r_balanced_tail<0, true>),
                   "opcode7_simt_32w3r_na_v4_fullunroll_balanced_tail_pdl_wait",
                   megakernel::lm_head_simt::grid_ctas,
                   megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED_PDL((megakernel::lm_head_simt::
                        rms_lm_head_32w3r_balanced_tail<1, true>),
                   "opcode7_simt_32w3r_na_v4_fullunroll_prefetch128_pdl_wait",
                   megakernel::lm_head_simt::grid_ctas,
                   megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED_PDL((megakernel::lm_head_simt::
                        rms_lm_head_32w3r_balanced_tail<2, true>),
                   "opcode7_simt_32w3r_na_v4_fullunroll_prefetch256_pdl_wait",
                   megakernel::lm_head_simt::grid_ctas,
                   megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED_PDL((megakernel::lm_head_simt::
                        rms_lm_head_32w3r_balanced_tail<
                            2, true,
                            megakernel::mlp_simt::prefetch_policy_l1>),
                   "opcode7_simt_32w3r_na_v4_fullunroll_prefetch256_l1_"
                   "pdl_wait",
                   megakernel::lm_head_simt::grid_ctas,
                   megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED_PDL((megakernel::lm_head_simt::
                        rms_lm_head_32w3r_balanced_tail<
                            2, true,
                            megakernel::mlp_simt::prefetch_policy_l2_evict_last>),
                   "opcode7_simt_32w3r_na_v4_fullunroll_prefetch256_"
                   "l2_evict_last_pdl_wait",
                   megakernel::lm_head_simt::grid_ctas,
                   megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED_PDL((megakernel::lm_head_simt::
                        rms_lm_head_32w3r_balanced_tail<4, true>),
                   "opcode7_simt_32w3r_na_v4_fullunroll_prefetch512_pdl_wait",
                   megakernel::lm_head_simt::grid_ctas,
                   megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED_PDL((megakernel::lm_head_simt::
                        rms_lm_head_32w3r_balanced_tail<8, true>),
                   "opcode7_simt_32w3r_na_v4_fullunroll_prefetch1024_pdl_wait",
                   megakernel::lm_head_simt::grid_ctas,
                   megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED_PDL((megakernel::lm_head_simt::
                        rms_lm_head_32w3r_balanced_tail<12, true>),
                   "opcode7_simt_32w3r_na_v4_fullunroll_prefetch1536_pdl_wait",
                   megakernel::lm_head_simt::grid_ctas,
                   megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED_PDL((megakernel::lm_head_simt::
                        rms_lm_head_32w3r_balanced_tail<16, true>),
                   "opcode7_simt_32w3r_na_v4_fullunroll_prefetch2048_pdl_wait",
                   megakernel::lm_head_simt::grid_ctas,
                   megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED_PDL((megakernel::lm_head_simt::
                        rms_lm_head_32w3r_balanced_tail<24, true>),
                   "opcode7_simt_32w3r_na_v4_fullunroll_prefetch3072_pdl_wait",
                   megakernel::lm_head_simt::grid_ctas,
                   megakernel::lm_head_simt::threads_per_cta);
    BIND_FIXED_PDL((megakernel::lm_head_simt::
                        rms_lm_head_32w3r_balanced_tail<32, true>),
                   "opcode7_simt_32w3r_na_v4_fullunroll_prefetch4096_pdl_wait",
                   megakernel::lm_head_simt::grid_ctas,
                   megakernel::lm_head_simt::threads_per_cta);
}

#undef BIND_FIXED
#undef BIND_FIXED_PDL
#undef BIND_FIXED_SMEM
#undef LLAMA_GLOBAL_MEMBERS
