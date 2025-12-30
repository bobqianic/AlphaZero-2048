//
// Created by qianp on 27/12/2025.
//

// model.cpp
#include "model.h"
#include <cmath>
#include <array>
#include <algorithm>

namespace rl2048 {

float muz_h(float x, float eps) {
  const float s  = (x >= 0.0f) ? 1.0f : -1.0f;
  const float ax = std::abs(x);
  return s * (std::sqrt(ax + 1.0f) - 1.0f) + eps * x;
}

float muz_h_inv(float y, float eps) {
  const float s  = (y >= 0.0f) ? 1.0f : -1.0f;
  const float ay = std::abs(y);
  const float inside = 1.0f + 4.0f * eps * (ay + 1.0f + eps);
  const float t = (std::sqrt(inside) - 1.0f) / (2.0f * eps);
  const float x = t * t - 1.0f;
  return s * x;
}

inline torch::Tensor muz_h_inv_tensor(const torch::Tensor& y, float eps) {
  // y: float tensor on CPU or CUDA
  auto abs_y = torch::abs(y);
  auto sign_y = torch::sign(y);

  // x = ((sqrt(1 + 4*eps*(abs(y)+1+eps)) - 1) / (2*eps))^2 - 1
  auto t = (torch::sqrt(1.0f + 4.0f * eps * (abs_y + 1.0f + eps)) - 1.0f) / (2.0f * eps);
  auto x = t * t - 1.0f;

  return sign_y * x;
}

// Writes into out[0..495]
void encode_board_31bit_inplace(env2048::Env::Board b, float* out) noexcept {
  // For each of 16 cells, write 31 floats
  for (int cell = 0; cell < 16; ++cell) {
    const std::uint8_t exp = static_cast<std::uint8_t>((b >> (cell * 4)) & 0xFULL);

    float* dst = out + cell * 31;

    // Zero all 31 floats (memset is safe for 0.0f)
    std::memset(dst, 0, 31 * sizeof(float));

    // exp==0 => empty tile => all zeros
    // otherwise set the "bit position" exp to 1.0
    // exp is 0..15 here (4-bit), so always within 0..30 range except exp==0 case.
    if (exp != 0 && exp < 31) {
      dst[exp] = 1.0f;
    }
  }
}

torch::Tensor scalar_to_support_dist(torch::Tensor y, int support_max) {
  // y: [B] float
  y = y.clamp(0.0, (double)support_max);

  auto low   = y.floor().to(torch::kLong);
  auto high  = (low + 1).clamp(0, support_max);

  auto y_low  = low.to(y.dtype());
  auto w_high = (y - y_low);
  auto w_low  = (1.0 - w_high);

  auto dist = torch::zeros({y.size(0), support_max + 1}, y.options());
  dist.scatter_add_(1, low.unsqueeze(1),  w_low.unsqueeze(1));
  dist.scatter_add_(1, high.unsqueeze(1), w_high.unsqueeze(1));
  return dist;
}

torch::Tensor decode_value_raw(torch::Tensor value_logits, const torch::Tensor& support, float eps) {
  // value_logits: [B,601] on same device as support
  auto prob = torch::softmax(value_logits, 1);       // [B,601]
  auto expected_y = (prob * support).sum(1);      // [B]
  return muz_h_inv_tensor(expected_y, eps);                     // [B], stays on device
}


// ---- ResBlock ----
ResBlockImpl::ResBlockImpl(int dim)
    : ln1(torch::nn::LayerNormOptions(std::vector<int64_t>{dim})),
      ln2(torch::nn::LayerNormOptions(std::vector<int64_t>{dim})),
      fc1(dim, dim),
      fc2(dim, dim) {
  register_module("ln1", ln1);
  register_module("ln2", ln2);
  register_module("fc1", fc1);
  register_module("fc2", fc2);
}

torch::Tensor ResBlockImpl::forward(torch::Tensor x) {
  auto y = ln1(x);
  y = torch::relu(y);
  y = fc1(y);
  y = ln2(y);
  y = torch::relu(y);
  y = fc2(y);
  return x + y;
}

// ---- Net ----
NetImpl::NetImpl()
    : upscale(kObsDim, kHiddenDim),
      ln_head(torch::nn::LayerNormOptions(std::vector<int64_t>{kHiddenDim})),
      policy(kHiddenDim, kNumActions),
      value(kHiddenDim, kSupportSize),
      reward(kHiddenDim, kSupportSize) {

  register_module("upscale", upscale);

  blocks.reserve(kNumBlocks);
  for (int i = 0; i < kNumBlocks; ++i) {
    blocks.push_back(ResBlock(kHiddenDim));
    register_module("block_" + std::to_string(i), blocks.back());
  }

  register_module("ln_head", ln_head);
  register_module("policy", policy);
  register_module("value", value);
  register_module("reward", reward);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> NetImpl::forward(torch::Tensor x) {
  if (x.dim() == 1) x = x.unsqueeze(0); // [1,496]

  auto h = upscale(x);
  for (auto& b : blocks) h = b->forward(h);

  auto z = ln_head(h);
  z = torch::relu(z);

  auto p = policy(z);
  auto v = value(z);
  auto r = reward(z);
  return {p, v, r};
}

} // namespace rl2048
