#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// CUDA forward kernel for linear layer
template <typename scalar_t>
__global__ void linear_forward_kernel(const scalar_t *__restrict__ input,
                                      const scalar_t *__restrict__ weight,
                                      const scalar_t *__restrict__ bias,
                                      scalar_t *__restrict__ output,
                                      int batch_size,
                                      int in_features,
                                      int out_features)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    if (idx < total)
    {
        int i = idx / out_features;
        int j = idx % out_features;
        scalar_t sum = bias[j];
        for (int k = 0; k < in_features; ++k)
        {
            sum += input[i * in_features + k] * weight[j * in_features + k];
        }
        output[i * out_features + j] = sum;
    }
}

torch::Tensor linear_forward_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias)
{
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);

    auto output = torch::zeros({batch_size, out_features}, input.options());
    int total = batch_size * out_features;
    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "linear_forward_cuda", ([&]
                                                                            { linear_forward_kernel<scalar_t><<<numBlocks, blockSize>>>(
                                                                                  input.data_ptr<scalar_t>(),
                                                                                  weight.data_ptr<scalar_t>(),
                                                                                  bias.data_ptr<scalar_t>(),
                                                                                  output.data_ptr<scalar_t>(),
                                                                                  batch_size,
                                                                                  in_features,
                                                                                  out_features); }));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &linear_forward_cuda, "Linear layer forward (CUDA)");
}
