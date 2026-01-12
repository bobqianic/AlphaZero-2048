// generator.h
#pragma once

#include "util/core/env2048.h"
#include "util/core/mcts.h"
#include "model.h"
#include "util/core/replaybuffer.h"
#include "util/core/step.h"
#include "util/logger.h"
#include "util/cuda/encode2048.h"
#include "util/core/mcts2048_common.h"

#include <torch/torch.h>


#include <cstdint>
#include <memory>


using mcts2048_common::S2048;
using mcts2048_common::DiscountedReturnBackup;
using mcts2048_common::BatchedEvaluator;
using mcts2048_common::TorchAsyncMCTSModel;
using mcts2048_common::splitmix64;
using mcts2048_common::sample_action;


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

    // if true, evaluate value on afterstates (chance nodes).
    bool afterstate = true;
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
