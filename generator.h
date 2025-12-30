// generator.h
#pragma once

#include "util/core/env2048.h"
#include "util/core/mcts.h"
#include "model.h"
#include "util/core/replaybuffer.h"
#include "util/core/step.h"
#include "util/logger.h"
#include "util/cuda/encode2048.h"

#include <torch/torch.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct GeneratorConfig {
    int episodes = 100;
    int sims = 100;
    int workers = 1;

    // MCTS params
    double cpuct = 1.5;
    double dirichlet_alpha = 0.25;
    double dirichlet_epsilon = 0.1;

    // Evaluator batching
    int eval_max_batch = 8192;
    int eval_max_wait_us = 500;

    // Keep only tail steps per episode (like your old disk writer)
    int save_tail = 200;

    // RNG
    std::uint64_t seed = 1;
};

// Temperature schedule driven by train_step (same behavior as your original)
double temperature_for_step(std::uint64_t train_step);

class Generator {
public:
    Generator(rl2048::Net net, torch::Device device, const GeneratorConfig& cfg, std::shared_ptr<Logger> logger);

    // Generate episodes with current net, push into replay (in-memory).
    void generate_into(ReplayBuffer& replay, std::uint64_t train_step);

private:
    rl2048::Net net_;
    torch::Device device_;
    GeneratorConfig cfg_;
    std::shared_ptr<Logger> logger_;
};
