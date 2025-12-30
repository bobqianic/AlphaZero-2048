// trainer.h
#pragma once

#include "util/core/replaybuffer.h"
#include "model.h"
#include "util/logger.h"

#include <torch/torch.h>

#include <cstdint>
#include <string>

struct TrainerConfig {
    int64_t batch = 1024;
    double lr = 3e-4;

    int64_t log_every = 100;
};

class Trainer {
public:
    Trainer(rl2048::Net net, torch::Device device, const TrainerConfig& cfg, std::shared_ptr<Logger> logger);

    rl2048::Net& net() { return net_; }
    torch::optim::Adam& optimizer() { return opt_; }

    // Train for `num_steps` gradient steps. Updates train_step counter.
    void train_steps(ReplayBuffer& replay, int64_t num_steps, std::uint64_t& train_step);

    // Save/load full checkpoint (model + optimizer + train_step)
    void save_checkpoint(const std::string& path, std::uint64_t train_step);
    bool load_checkpoint(const std::string& path, std::uint64_t& train_step);

private:
    std::shared_ptr<Logger> logger_;

    rl2048::Net net_;
    torch::Device device_;
    TrainerConfig cfg_;
    torch::optim::Adam opt_;

    // Support tensor for decoding value distribution -> scalar (on device)
    torch::Tensor support_;

    // CPU (possibly pinned) staging buffers, reused each step
    torch::Tensor xb_cpu_;   // [B, obs_dim]
    torch::Tensor pib_cpu_;  // [B, 4]
    torch::Tensor rt_cpu_;   // [B]
    torch::Tensor vt_cpu_;   // [B]
    torch::Tensor iw_cpu_;   // [B]

    torch::Tensor yv_cpu_;   // [B] muz_h(vt)
    torch::Tensor yr_cpu_;   // [B] muz_h(rt)

    torch::TensorOptions cpu_f32_;

    void ensure_cpu_buffers_();
    void apply_muz_h_cpu_(const torch::Tensor& src_cpu, torch::Tensor& dst_cpu);
};
