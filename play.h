//
// Created by qianp on 29/12/2025.
//

#pragma once

#include <cstdint>
#include <iosfwd>
#include <torch/torch.h>

#include "model.h"
#include "util/core/env2048.h"

namespace play2048 {

    struct PlayConfig {
        // MCTS
        int sims = 800;
        double cpuct = 1.5;
        double dirichlet_alpha = 0.0;     // set >0 if you want root noise
        double dirichlet_epsilon = 0.0;
        double temperature = 0.0;         // 0 = argmax(visits), >0 = sample from visit dist

        // Batched evaluator
        int eval_max_batch = 64;
        int eval_max_wait_us = 200;

        // Search pumping
        int sim_budget_per_pump = 16;
        int max_inflight_per_task = 256;

        // Game
        std::uint64_t seed = 0xC0FFEEULL;
        int initial_tiles = 2;
        int max_steps = 1000000;          // safety
        bool print_each_step = true;
    };

    struct PlayResult {
        std::uint32_t total_reward = 0;
        std::uint32_t steps = 0;
        std::uint32_t max_tile = 0;
        env2048::Env::Board final_board = 0;
    };

    void print_board(env2048::Env::Board b, std::ostream& os);
    PlayResult play(rl2048::Net net, torch::Device device, const PlayConfig& cfg, std::ostream& os);

} // namespace play2048
