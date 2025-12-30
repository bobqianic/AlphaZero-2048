#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAException.h>
#include "c10/cuda/CUDAStream.h"

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE(x, t) TORCH_CHECK((x).scalar_type() == (t), #x " has wrong dtype")

__global__ void encode2048_onehot31_kernel(
    const int64_t* __restrict__ boards,
    float* __restrict__ out,
    int64_t B
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = B * 16;
    if (idx >= n) return;

    int64_t i = idx >> 4;
    int cell  = (int)(idx & 15);

    uint64_t b = (uint64_t)boards[i];
    uint8_t exp = (uint8_t)((b >> (cell * 4)) & 0xFULL);

    if (exp != 0 && exp < 31) {
        out[idx * 31 + exp] = 1.0f;
    }
}

void encode2048_onehot31_out_cuda(torch::Tensor boards, torch::Tensor out) {
    CHECK_CUDA(boards);
    CHECK_CUDA(out);
    CHECK_CONTIGUOUS(boards);
    CHECK_CONTIGUOUS(out);
    CHECK_DTYPE(boards, torch::kInt64);
    CHECK_DTYPE(out, torch::kFloat32);

    TORCH_CHECK(boards.dim() == 1, "boards must be [B]");
    TORCH_CHECK(out.dim() == 2, "out must be [B, 496]");
    TORCH_CHECK(out.size(0) == boards.size(0), "B mismatch");
    TORCH_CHECK(out.size(1) == 16 * 31, "out must have 496 cols");

    int64_t B = boards.size(0);
    if (B == 0) return;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(boards.device().index());

    size_t bytes = (size_t)B * (size_t)(16 * 31) * sizeof(float);
    C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr<float>(), 0, bytes, stream));

    int threads = 256;
    int64_t nthreads = B * 16;
    int blocks = (int)((nthreads + threads - 1) / threads);

    encode2048_onehot31_kernel<<<blocks, threads, 0, stream>>>(
        boards.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        B
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
