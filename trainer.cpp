// trainer.cpp
// Trainer for 2048 policy/value/reward network using libtorch.
// Loads episode files from --data_dir, trains with PER + Adam.
//
// Fixes:
//  - No host writes into CUDA tensors (the cause of your SIGSEGV).
//  - Encode batch on CPU via encode_board_31bit_inplace(), then upload once.

// trainer.cpp
#include "trainer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <vector>

#include <ATen/autocast_mode.h>

namespace fs = std::filesystem;

struct AutocastGuard {
  at::DeviceType device_type;
  bool prev_enabled;
  at::ScalarType prev_dtype;

  AutocastGuard(at::DeviceType dt, at::ScalarType dtype, bool enabled = true)
      : device_type(dt),
        prev_enabled(at::autocast::is_autocast_enabled(dt)),
        prev_dtype(at::autocast::get_autocast_dtype(dt)) {
    at::autocast::set_autocast_enabled(dt, enabled);
    at::autocast::set_autocast_dtype(dt, dtype);
  }

  ~AutocastGuard() {
    at::autocast::set_autocast_dtype(device_type, prev_dtype);
    at::autocast::set_autocast_enabled(device_type, prev_enabled);
  }
};

Trainer::Trainer(rl2048::Net net, torch::Device device, const TrainerConfig& cfg, std::shared_ptr<Logger> logger)
  : net_(std::move(net)),
    device_(device),
    cfg_(cfg),
    logger_(std::move(logger)),
    opt_(net_->parameters(), torch::optim::AdamOptions(cfg_.lr)) {

  net_->to(device_);
  net_->train();

  support_ = torch::arange(
    0, rl2048::kSupportMax + 1,
    torch::TensorOptions().dtype(torch::kFloat32).device(device_)
  ).view({1, -1}); // [1, 601]

  cpu_f32_ = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  if (device_.is_cuda()) cpu_f32_ = cpu_f32_.pinned_memory(true);

  ensure_cpu_buffers_();
}

void Trainer::ensure_cpu_buffers_() {
  const int64_t B = cfg_.batch;

  if (xb_cpu_.defined() && xb_cpu_.sizes() == torch::IntArrayRef({B, (int64_t)rl2048::kObsDim})) return;

  xb_cpu_  = torch::empty({B, (int64_t)rl2048::kObsDim}, cpu_f32_);
  pib_cpu_ = torch::empty({B, 4}, cpu_f32_);
  rt_cpu_  = torch::empty({B}, cpu_f32_);
  vt_cpu_  = torch::empty({B}, cpu_f32_);
  iw_cpu_  = torch::empty({B}, cpu_f32_);

  yv_cpu_  = torch::empty({B}, cpu_f32_);
  yr_cpu_  = torch::empty({B}, cpu_f32_);
}

void Trainer::apply_muz_h_cpu_(const torch::Tensor& src_cpu, torch::Tensor& dst_cpu) {
  // src_cpu/dst_cpu are [B] on CPU (possibly pinned). Compute elementwise muz_h using your helper.
  const int64_t n = src_cpu.numel();
  const float* sp = src_cpu.data_ptr<float>();
  float* dp = dst_cpu.data_ptr<float>();
  for (int64_t i = 0; i < n; ++i) dp[i] = rl2048::muz_h(sp[i]);
}

void Trainer::train_steps(ReplayBuffer& replay, int64_t num_steps, std::uint64_t& train_step) {
  ensure_cpu_buffers_();

  std::mt19937_64 rng(0xC0FFEEULL ^ (std::uint64_t)train_step);

  std::vector<Step> batch_steps;
  std::vector<int> batch_indices;
  std::vector<float> batch_isw;

  for (int64_t step = 1; step <= num_steps; ++step) {
    // ---- Sample batch from replay (CPU) ----
    replay.sample_batch((int)cfg_.batch, rng, batch_steps, batch_indices, batch_isw);
    if ((int64_t)batch_steps.size() != cfg_.batch) {
      // stop requested or insufficient data; just return
      return;
    }

    // ---- Fill CPU staging tensors ----
    float* xb_ptr  = xb_cpu_.data_ptr<float>();
    float* pib_ptr = pib_cpu_.data_ptr<float>();
    float* rt_ptr  = rt_cpu_.data_ptr<float>();
    float* vt_ptr  = vt_cpu_.data_ptr<float>();
    float* iw_ptr  = iw_cpu_.data_ptr<float>();

    for (int64_t b = 0; b < cfg_.batch; ++b) {
      const Step& s = batch_steps[(size_t)b];

      rl2048::encode_board_31bit_inplace(s.board, xb_ptr + b * (int64_t)rl2048::kObsDim);

      pib_ptr[b * 4 + 0] = s.pi[0];
      pib_ptr[b * 4 + 1] = s.pi[1];
      pib_ptr[b * 4 + 2] = s.pi[2];
      pib_ptr[b * 4 + 3] = s.pi[3];

      rt_ptr[b] = (float)s.reward;
      vt_ptr[b] = (float)s.value_target;

      iw_ptr[b] = batch_isw[(size_t)b];
    }

    // ---- Upload once to device ----
    auto xb  = xb_cpu_.to(device_, /*non_blocking=*/device_.is_cuda());
    auto pib = pib_cpu_.to(device_, /*non_blocking=*/device_.is_cuda());
    auto rt  = rt_cpu_.to(device_, /*non_blocking=*/device_.is_cuda());
    auto vt  = vt_cpu_.to(device_, /*non_blocking=*/device_.is_cuda());
    auto iw  = iw_cpu_.to(device_, /*non_blocking=*/device_.is_cuda());

    // ---- Forward ----
    torch::Tensor policy_logits, value_logits, reward_logits;
    {
        AutocastGuard amp(at::kCUDA, at::kBFloat16, true);
        std::tie(policy_logits, value_logits, reward_logits) = net_->forward(xb);
    }
    //policy_logits  [B,4]
    //value_logits   [B,601]
    //reward_logits  [B,601]

    // Policy loss per-sample: -sum(pi * log_softmax)
    auto logp = torch::log_softmax(policy_logits.to(torch::kFloat32), 1);
    auto policy_loss_per = -(pib * logp).sum(1); // [B]

    // ---- Value target distribution ----
    apply_muz_h_cpu_(vt_cpu_, yv_cpu_);
    auto yv_dev = yv_cpu_.to(device_, /*non_blocking=*/device_.is_cuda());

    auto v_dist = rl2048::scalar_to_support_dist(yv_dev, rl2048::kSupportMax); // [B,601]
    auto logv = torch::log_softmax(value_logits.to(torch::kFloat32), 1);
    auto value_loss_per = -(v_dist * logv).sum(1); // [B]

    // ---- Reward target distribution ----
    apply_muz_h_cpu_(rt_cpu_, yr_cpu_);
    auto yr_dev = yr_cpu_.to(device_, /*non_blocking=*/device_.is_cuda());

    auto r_dist = rl2048::scalar_to_support_dist(yr_dev, rl2048::kSupportMax); // [B,601]
    auto logr = torch::log_softmax(reward_logits.to(torch::kFloat32), 1);
    auto reward_loss_per = -(r_dist * logr).sum(1); // [B]

    // Total per-sample loss with importance weights
    auto total_loss_per = policy_loss_per + value_loss_per + reward_loss_per;
    auto loss = (total_loss_per * iw).mean();

    opt_.zero_grad();
    loss.backward();
    opt_.step();

    // ---- Update priorities (PER) ----
    {
      torch::NoGradGuard ng;

      auto pred_v_raw = rl2048::decode_value_raw(value_logits, support_, rl2048::kEpsTransform); // [B]
      auto err_cpu = (pred_v_raw - vt).abs().add(1e-6).to(torch::kCPU).contiguous();

      std::vector<float> new_p((size_t)cfg_.batch);
      std::memcpy(new_p.data(), err_cpu.data_ptr<float>(), (size_t)cfg_.batch * sizeof(float));

      replay.update_priorities(batch_indices, new_p);
    }

    // Logging
    if (cfg_.log_every > 0 && (step % cfg_.log_every) == 0) {
      logf(*logger_,
           "train_step=", train_step,
           " step_in_call=", step,
           " loss=", loss.item<double>(),
           " (policy=", policy_loss_per.mean().item<double>(),
           " value=",  value_loss_per.mean().item<double>(),
           " reward=", reward_loss_per.mean().item<double>(),
           ")");
    }

    ++train_step;
  }
}

void Trainer::save_checkpoint(const std::string& path, std::uint64_t train_step) {
  torch::serialize::OutputArchive arch;
  net_->save(arch);
  opt_.save(arch);

  auto ts = torch::tensor((int64_t)train_step, torch::TensorOptions().dtype(torch::kInt64));
  arch.write("train_step", ts);

  arch.save_to(path);
  logf(*logger_, "Saved checkpoint: ", path, " (train_step=", train_step, ")");
}

bool Trainer::load_checkpoint(const std::string& path, std::uint64_t& train_step) {
  if (!fs::exists(path)) return false;

  torch::serialize::InputArchive arch;
  arch.load_from(path);

  net_->load(arch);
  opt_.load(arch);

  torch::Tensor ts;
  if (arch.try_read("train_step", ts)) {
    train_step = (std::uint64_t)ts.item<int64_t>();
  } else {
    train_step = 0;
  }

  net_->to(device_);
  net_->train();

  logf(*logger_, "Loaded checkpoint: ", path, " (train_step=", train_step, ")");
  return true;
}
