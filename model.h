//
// Created by qianp on 27/12/2025.
//

// model.h
#pragma once

#include "util/core/env2048.h"
#include <torch/torch.h>
#include <cstdint>
#include <tuple>
#include <vector>

namespace rl2048 {

    // ---- Constants ----
    static constexpr int   kObsDim     = 496;   // 16 * 31 bits
    static constexpr int   kHiddenDim  = 256;
    static constexpr int   kNumBlocks  = 10;
    static constexpr int   kNumActions = 4;

    static constexpr int   kSupportMax  = 600;
    static constexpr int   kSupportSize = kSupportMax + 1;

    static constexpr float kEpsTransform = 0.001f;

    // ---- MuZero transform h and inverse ----
    float muz_h(float x, float eps = kEpsTransform);
    float muz_h_inv(float y, float eps = kEpsTransform);

    // ---- Encoding: 31-bit binary per tile => 496 floats ----
    void encode_board_31bit_inplace(env2048::Env::Board b, float* out) noexcept;

    // ---- Support helpers ----
    // y should be already transformed (h(x)). Clamps to [0..support_max] and linearly interpolates.
    torch::Tensor scalar_to_support_dist(torch::Tensor y, int support_max = kSupportMax);

    // Decode value logits -> expected support -> inverse transform -> raw scalar return.
    // value_logits: [B, 601] -> returns [B] on same device.
    // support: [1, 601]
    torch::Tensor decode_value_raw(torch::Tensor value_logits, const torch::Tensor& support, float eps);

    // ---- Network ----
    struct ResBlockImpl : torch::nn::Module {
        torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
        torch::nn::Linear fc1{nullptr}, fc2{nullptr};

        explicit ResBlockImpl(int dim = kHiddenDim);
        torch::Tensor forward(torch::Tensor x);
    };
    TORCH_MODULE(ResBlock);

    struct NetImpl : torch::nn::Module {
        torch::nn::Linear upscale{nullptr};
        std::vector<ResBlock> blocks;

        torch::nn::LayerNorm ln_head{nullptr};
        torch::nn::Linear policy{nullptr}, value{nullptr}, reward{nullptr};

        NetImpl();

        // Returns: (policy_logits [B,4], value_logits [B,601], reward_logits [B,601])
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    };
    TORCH_MODULE(Net);

} // namespace rl2048
