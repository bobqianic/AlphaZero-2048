#pragma once
#include <torch/torch.h>

// boards_dev_int64: [B] CUDA int64
// out_dev_f32:      [B, 496] CUDA float32
void encode2048_onehot31_out_cuda(torch::Tensor boards_dev_int64,
                                 torch::Tensor out_dev_f32);
