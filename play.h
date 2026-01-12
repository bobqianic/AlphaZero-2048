//
// Created by qianp on 29/12/2025.
//

#pragma once

#pragma once

#include <cstdint>
#include <iosfwd>
#include <vector>
#include <torch/torch.h>

#include "model.h"
#include "util/core/env2048.h"
#include "util/core/mcts2048_common.h"
#include "util/core/mcts.h"
#include "util/logger.h"

using mcts2048_common::S2048;
using mcts2048_common::DiscountedReturnBackup;
using mcts2048_common::BatchedEvaluator;
using mcts2048_common::TorchAsyncMCTSModel;
using mcts2048_common::splitmix64;
using mcts2048_common::sample_action;

namespace play2048 {

    struct PlayConfig {
        // (whatever you already have)
        std::uint64_t seed = 1;
        int sims = 256;
        double cpuct = 1.5;
        double dirichlet_alpha = 0.0;
        double dirichlet_epsilon = 0.0;
        double temperature = 0.0;

        int max_steps = 100000;
        int initial_tiles = 2;
        bool print_each_step = false;

        // evaluator tuning
        int eval_max_batch = 8192;
        int eval_max_wait_us = 500;

        // mcts_task tuning
        int max_inflight_per_task = 0;      // 0 => auto
        int sim_budget_per_pump = 16;       // pump granularity (single-game play())

        // if true, evaluate value on afterstates (chance nodes).
        bool afterstate = true;
    };

    struct PlayResult {
        std::uint64_t final_board = 0;
        std::uint32_t total_reward = 0;
        std::uint32_t max_tile = 0;
        std::uint32_t steps = 0;
    };

    void print_board(env2048::Env::Board b, std::ostream& os);
    PlayResult play(rl2048::Net net, torch::Device device, const PlayConfig& cfg, std::ostream& os);

    // multi-game
    struct MultiPlayConfig {
        PlayConfig game;     // per-game config (sims/cpuct/temp/etc.)
        int games = 64;      // number of concurrent games
        int workers = 0;     // 0 => std::thread::hardware_concurrency()
        int sim_budget = 16; // how many sims each pump() does before re-queue

        // printing control (printing all games is usually unreadable)
        int print_game = -1; // -1 => none, else print that game id
        bool print_initial_board = false;
    };

    struct TileStat {
        std::uint32_t tile = 0;   // e.g. 128, 256, ...
        std::uint32_t count = 0;  // number of games
        double pct = 0.0;         // percent of games in [0..100]
    };

    struct MultiPlayResult {
        std::vector<PlayResult> results;

        // simple aggregates
        double avg_total_reward = 0.0;
        double avg_steps = 0.0;
        std::uint32_t best_max_tile = 0;

        // max-tile statistics across games
        // - exact:    max tile == X
        // - at_least: max tile >= X  (reach rate)
        std::vector<TileStat> max_tile_exact;
        std::vector<TileStat> max_tile_at_least;
    };

    MultiPlayResult play_many(rl2048::Net net, torch::Device device,
                              const MultiPlayConfig& cfg, std::ostream& os);

} // namespace play2048
