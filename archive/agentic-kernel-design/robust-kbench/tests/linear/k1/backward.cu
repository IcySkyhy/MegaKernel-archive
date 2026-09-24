#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// CUDA kernel: gradient with respect to input
template <typename scalar_t>
__global__ void linear_backward_input_kernel(const scalar_t *__restrict__ grad_output,
                                             const scalar_t *__restrict__ weight,
                                             scalar_t *__restrict__ grad_input,
                                             int batch_size,
                                             int in_features,
                                             int out_features)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * in_features;
    if (idx < total)
    {
        int i = idx / in_features;
        int k = idx % in_features;
        scalar_t sum = 0;
        for (int j = 0; j < out_features; ++j)
        {
            sum += grad_output[i * out_features + j] * weight[j * in_features + k];
        }
        grad_input[i * in_features + k] = sum;
    }
}

// CUDA kernel: gradient with respect to weight
template <typename scalar_t>
__global__ void linear_backward_weight_kernel(const scalar_t *__restrict__ grad_output,
                                              const scalar_t *__restrict__ input,
                                              scalar_t *__restrict__ grad_weight,
                                              int batch_size,
                                              int in_features,
                                              int out_features)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_features * in_features;
    if (idx < total)
    {
        int j = idx / in_features;
        int k = idx % in_features;
        scalar_t sum = 0;
        for (int i = 0; i < batch_size; ++i)
        {
            sum += grad_output[i * out_features + j] * input[i * in_features + k];
        }
        grad_weight[j * in_features + k] = sum;
    }
}

// CUDA kernel: gradient with respect to bias
template <typename scalar_t>
__global__ void linear_backward_bias_kernel(const scalar_t *__restrict__ grad_output,
                                            scalar_t *__restrict__ grad_bias,
                                            int batch_size,
                                            int out_features)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < out_features)
    {
        scalar_t sum = 0;
        for (int i = 0; i < batch_size; ++i)
        {
            sum += grad_output[i * out_features + j];
        }
        grad_bias[j] = sum;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
linear_backward_cuda(torch::Tensor grad_output,
                     torch::Tensor input,
                     torch::Tensor weight)
{
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto grad_input = torch::zeros_like(input);
    auto grad_weight = torch::zeros_like(weight);
    auto grad_bias = torch::zeros({out_features}, input.options());

    int blockSize = 256;

    // grad_input
    int total_input = batch_size * in_features;
    int numBlocks = (total_input + blockSize - 1) / blockSize;
    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "linear_backward_input_cuda", ([&]
                                                                                         { linear_backward_input_kernel<scalar_t><<<numBlocks, blockSize>>>(
                                                                                               grad_output.data_ptr<scalar_t>(),
                                                                                               weight.data_ptr<scalar_t>(),
                                                                                               grad_input.data_ptr<scalar_t>(),
                                                                                               batch_size,
                                                                                               in_features,
                                                                                               out_features); }));

    // grad_weight
    int total_weight = out_features * in_features;
    numBlocks = (total_weight + blockSize - 1) / blockSize;
    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "linear_backward_weight_cuda", ([&]
                                                                                          { linear_backward_weight_kernel<scalar_t><<<numBlocks, blockSize>>>(
                                                                                                grad_output.data_ptr<scalar_t>(),
                                                                                                input.data_ptr<scalar_t>(),
                                                                                                grad_weight.data_ptr<scalar_t>(),
                                                                                                batch_size,
                                                                                                in_features,
                                                                                                out_features); }));

    // grad_bias
    numBlocks = (out_features + blockSize - 1) / blockSize;
    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "linear_backward_bias_cuda", ([&]
                                                                                        { linear_backward_bias_kernel<scalar_t><<<numBlocks, blockSize>>>(
                                                                                              grad_output.data_ptr<scalar_t>(),
                                                                                              grad_bias.data_ptr<scalar_t>(),
                                                                                              batch_size,
                                                                                              out_features); }));

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("backward", &linear_backward_cuda, "Linear layer backward (CUDA)");
}
