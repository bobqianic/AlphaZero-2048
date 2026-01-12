// play.cpp
//
// Minimal “play a game” driver built to mirror generator.cpp’s MCTS+batched-eval flow,
// but for a single episode with board printing.
//
// + Added play_many(): play multiple games simultaneously sharing one BatchedEvaluator.
//

#include "play.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace play2048 {

static const char* action_name(env2048::Action a) {
  switch (a) {
    case env2048::Action::Up:    return "Up";
    case env2048::Action::Right: return "Right";
    case env2048::Action::Down:  return "Down";
    case env2048::Action::Left:  return "Left";
    default: return "?";
  }
}

void print_board(env2048::Env::Board b, std::ostream& os) {
  env2048::Env tmp(b);
  auto exps = tmp.to_exponents();
  os << "+------+------+------+------+\n";
  for (int r = 0; r < 4; ++r) {
    os << "|";
    for (int c = 0; c < 4; ++c) {
      const std::uint8_t e = exps[r * 4 + c];
      const std::uint32_t v = e ? (1u << e) : 0u;
      os << std::setw(6) << v << "|";
    }
    os << "\n+------+------+------+------+\n";
  }
}

static env2048::Action select_action_mcts(
    const env2048::Env& env,
    std::uint64_t episode_seed,
    std::uint64_t step_index,
    env2048::RNG& sampling_rng,
    TorchAsyncMCTSModel& model,
    const std::shared_ptr<BatchedEvaluator>& be,
    const PlayConfig& pcfg) {

  agmcts::MCTSConfig cfg;
  cfg.num_simulations = pcfg.sims;
  cfg.cpuct = pcfg.cpuct;
  cfg.dirichlet_alpha = pcfg.dirichlet_alpha;
  cfg.dirichlet_epsilon = pcfg.dirichlet_epsilon;
  cfg.temperature = pcfg.temperature;
  cfg.seed = splitmix64(pcfg.seed ^ episode_seed ^ (0x9E3779B97F4A7C15ull + step_index));
  cfg.use_afterstate_value = pcfg.afterstate;

  using Backup = DiscountedReturnBackup<S2048, agmcts::NoPayload>;
  using SearchTask = agmcts::mcts_task<S2048, TorchAsyncMCTSModel, agmcts::NoPayload, Backup>;

  const auto board = env.board();
  const std::uint64_t root_seed =
      splitmix64(episode_seed ^ (std::uint64_t)board ^ (0xA5A5A5A5A5A5A5A5ull + step_index));
  (void)root_seed;

  S2048 root(board);
  root.reward_from_parent_raw = 0.0;

  SearchTask search(cfg, &model, root, std::max(32, pcfg.max_inflight_per_task));

  while (true) {
    auto st = search.pump(std::max(1, pcfg.sim_budget_per_pump));
    if (st == SearchTask::PumpStatus::Waiting) {
      const uint64_t last = be->epoch();
      be->wait_for_epoch_change(last, /*timeout_us=*/200);
      continue;
    }
    if (st == SearchTask::PumpStatus::Done) break;
  }

  auto res = search.result();

  std::array<float,4> pi = {0,0,0,0};

  if (pcfg.temperature <= 1e-12) {
    int best_a = 0, best_v = -1;
    for (const auto& e : res.root_entries) {
      if (e.move.k != S2048::Move::Act) continue;
      if (e.visits > best_v) { best_v = e.visits; best_a = (int)e.move.v; }
    }
    pi[best_a] = 1.0f;
  } else {
    auto dist = res.visit_distribution(pcfg.temperature);
    for (auto& [mv, p] : dist) {
      if (mv.k != S2048::Move::Act) continue;
      pi[mv.v] = (float)p;
    }
  }

  if (pcfg.temperature <= 1e-12) {
    int best = 0;
    for (int a = 1; a < 4; ++a) if (pi[a] > pi[best]) best = a;
    return static_cast<env2048::Action>(best);
  }
  return sample_action(pi, sampling_rng);
}

// ---------------------------
// Main play() API (single game)
// ---------------------------
PlayResult play(rl2048::Net net, torch::Device device, const PlayConfig& cfg, std::ostream& os) {
  if (!net) throw std::runtime_error("play2048::play: net is null");

  net->to(device);
  net->eval();

  auto be = std::make_shared<BatchedEvaluator>(
      net, device,
      /*max_batch=*/cfg.eval_max_batch,
      /*max_wait_us=*/cfg.eval_max_wait_us);

  TorchAsyncMCTSModel amodel(be);

  // Episode seeding
  const std::uint64_t episode_seed = splitmix64(cfg.seed ^ 0xC3C3C3C3C3C3C3C3ull);

  env2048::RNG game_rng(splitmix64(episode_seed ^ 0x1111111111111111ull));     // for env.step spawns
  env2048::RNG sample_rng(splitmix64(episode_seed ^ 0x2222222222222222ull));   // for temperature sampling

  env2048::Env env;
  env.reset(game_rng, cfg.initial_tiles);

  PlayResult pr{};
  pr.final_board = env.board();
  pr.max_tile = env.max_tile();

  os << "=== 2048 play (MCTS) ===\n";
  os << "seed=" << cfg.seed << " sims=" << cfg.sims
     << " cpuct=" << cfg.cpuct << " temp=" << cfg.temperature << "\n\n";
  os << "Initial board:\n";
  print_board(env.board(), os);

  for (std::uint64_t step = 0; step < (std::uint64_t)cfg.max_steps; ++step) {
    if (env.is_terminal()) break;

    const auto action = select_action_mcts(env, episode_seed, step, sample_rng, amodel, be, cfg);

    auto sr = env.step(action, game_rng);

    // (Shouldn’t happen often; MCTS should avoid illegal moves, but be safe)
    if (!sr.moved) {
      // fall back: pick any legal move deterministically
      const std::uint8_t mask = env.legal_actions_mask();
      for (int a = 0; a < env2048::kNumActions; ++a) {
        if (mask & (1u << a)) { env.step(static_cast<env2048::Action>(a), game_rng); break; }
      }
    }

    pr.steps += 1;
    pr.total_reward += sr.reward;
    pr.max_tile = std::max(pr.max_tile, env.max_tile());
    pr.final_board = env.board();

    if (cfg.print_each_step) {
      os << "\nStep " << pr.steps
         << "  action=" << action_name(action)
         << "  reward=" << sr.reward
         << "  score=" << env.score()
         << "  max_tile=" << pr.max_tile
         << "\n";
      print_board(env.board(), os);
    }
  }

  os << "\n=== Game Over ===\n";
  os << "steps=" << pr.steps
     << " total_reward=" << pr.total_reward
     << " score=" << env.score()
     << " max_tile=" << pr.max_tile
     << "\n";
  return pr;
}

// ============================================================================
// play_many() — multi-game concurrent play sharing the same evaluator
// Mirrors generator.cpp’s WorkQueue + worker pool pattern.
// ============================================================================

namespace {

class WorkQueue {
public:
  void push(std::unique_ptr<struct WorkItem> w) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (stop_) return;
      q_.push_back(std::move(w));
    }
    cv_.notify_one();
  }

  std::unique_ptr<struct WorkItem> pop() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&]{ return stop_ || !q_.empty(); });
    if (q_.empty()) return nullptr;
    auto w = std::move(q_.front());
    q_.pop_front();
    return w;
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
  }

private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::unique_ptr<struct WorkItem>> q_;
  bool stop_ = false;
};

enum class WorkOutcome { RequeueProgress, RequeueWaiting, Finished };

struct WorkItem {
  virtual ~WorkItem() = default;
  virtual WorkOutcome pump(int sim_budget) = 0;
};

struct EpisodeCtx {
  int id = 0;
  std::uint64_t ep_seed = 0;
  std::uint64_t episode_seed = 0;

  env2048::RNG game_rng{0};
  env2048::RNG sample_rng{0};
  env2048::Env env;

  PlayResult pr{};
};

} // namespace

MultiPlayResult play_many(rl2048::Net net, torch::Device device,
                          const MultiPlayConfig& mcfg, std::ostream& os) {
  if (!net) throw std::runtime_error("play2048::play_many: net is null");

  MultiPlayResult out;
  if (mcfg.games <= 0) return out;

  net->to(device);
  net->eval();

  // Shared evaluator across ALL games
  auto be = std::make_shared<BatchedEvaluator>(
      net, device,
      /*max_batch=*/mcfg.game.eval_max_batch,
      /*max_wait_us=*/mcfg.game.eval_max_wait_us);

  TorchAsyncMCTSModel amodel(be);

  using Backup = DiscountedReturnBackup<S2048, agmcts::NoPayload>;
  using SearchTask = agmcts::mcts_task<S2048, TorchAsyncMCTSModel, agmcts::NoPayload, Backup>;

  // Base MCTS config (per-step seed overridden)
  agmcts::MCTSConfig cfg_base;
  cfg_base.num_simulations = mcfg.game.sims;
  cfg_base.cpuct = mcfg.game.cpuct;
  cfg_base.dirichlet_alpha = mcfg.game.dirichlet_alpha;
  cfg_base.dirichlet_epsilon = mcfg.game.dirichlet_epsilon;
  cfg_base.temperature = mcfg.game.temperature;
  cfg_base.seed = 0xC0FFEEULL ^ mcfg.game.seed;
  cfg_base.use_afterstate_value = mcfg.game.afterstate;

  const int max_inflight_per_task =
      std::max(32, mcfg.game.max_inflight_per_task);

  WorkQueue wq;
  std::atomic<int> done_games{0};

  out.results.resize((size_t)mcfg.games);

  std::mutex os_mu;

  struct StepSearchWork final : WorkItem {
    WorkQueue* wq = nullptr;
    TorchAsyncMCTSModel* model = nullptr;
    std::shared_ptr<BatchedEvaluator> be;

    agmcts::MCTSConfig cfg_base;
    int max_inflight_per_task = 0;

    const MultiPlayConfig* mcfg = nullptr;

    std::atomic<int>* done_games = nullptr;
    int total_games = 0;

    std::shared_ptr<EpisodeCtx> ep;
    std::vector<PlayResult>* results = nullptr;

    std::mutex* os_mu = nullptr;
    std::ostream* os = nullptr;

    std::unique_ptr<SearchTask> search;

    StepSearchWork(WorkQueue* wq_,
                   TorchAsyncMCTSModel* model_,
                   std::shared_ptr<BatchedEvaluator> be_,
                   const agmcts::MCTSConfig& cfg_base_,
                   int max_inflight_per_task_,
                   const MultiPlayConfig* mcfg_,
                   std::atomic<int>* done_games_,
                   int total_games_,
                   std::shared_ptr<EpisodeCtx> ep_,
                   std::vector<PlayResult>* results_,
                   std::mutex* os_mu_,
                   std::ostream* os_)
      : wq(wq_), model(model_), be(std::move(be_)),
        cfg_base(cfg_base_), max_inflight_per_task(max_inflight_per_task_),
        mcfg(mcfg_),
        done_games(done_games_), total_games(total_games_),
        ep(std::move(ep_)), results(results_),
        os_mu(os_mu_), os(os_) {

      const auto board = ep->env.board();

      agmcts::MCTSConfig cfg_step = cfg_base;
      cfg_step.seed = splitmix64(cfg_base.seed ^ ep->ep_seed ^ (std::uint64_t)ep->pr.steps);

      S2048 root(board);
      root.reward_from_parent_raw = 0.0;

      search = std::make_unique<SearchTask>(cfg_step, model, root, max_inflight_per_task);
    }

    WorkOutcome pump(int sim_budget) override {
      auto st = search->pump(std::max(1, sim_budget));

      if (st == SearchTask::PumpStatus::Waiting) {
        const uint64_t last = be->epoch();
        be->wait_for_epoch_change(last, /*timeout_us=*/200);
        return WorkOutcome::RequeueWaiting;
      }

      if (st != SearchTask::PumpStatus::Done) {
        return WorkOutcome::RequeueProgress;
      }

      // Pick action from root visits
      auto res = search->result();

      std::array<float,4> pi = {0,0,0,0};
      const double temp = mcfg->game.temperature;

      if (temp <= 1e-12) {
        int best_a = 0, best_v = -1;
        for (const auto& e : res.root_entries) {
          if (e.move.k != S2048::Move::Act) continue;
          if (e.visits > best_v) { best_v = e.visits; best_a = (int)e.move.v; }
        }
        pi[best_a] = 1.0f;
      } else {
        auto dist = res.visit_distribution(temp);
        for (auto& [mv, p] : dist) {
          if (mv.k != S2048::Move::Act) continue;
          pi[mv.v] = (float)p;
        }
      }

      env2048::Action action;
      if (temp <= 1e-12) {
        int best = 0;
        for (int a = 1; a < 4; ++a) if (pi[a] > pi[best]) best = a;
        action = static_cast<env2048::Action>(best);
      } else {
        action = sample_action(pi, ep->sample_rng);
      }

      // Step environment
      auto sr = ep->env.step(action, ep->game_rng);
      if (!sr.moved) {
        // defensive fallback
        const std::uint8_t mask = ep->env.legal_actions_mask();
        for (int a = 0; a < env2048::kNumActions; ++a) {
          if (mask & (1u << a)) { ep->env.step((env2048::Action)a, ep->game_rng); break; }
        }
      }

      ep->pr.steps += 1;
      ep->pr.total_reward += sr.reward;
      ep->pr.max_tile = std::max(ep->pr.max_tile, ep->env.max_tile());
      ep->pr.final_board = ep->env.board();

      const bool should_print =
        (mcfg->print_game >= 0 && ep->id == mcfg->print_game) && mcfg->game.print_each_step;

      if (should_print) {
        std::lock_guard<std::mutex> lk(*os_mu);
        (*os) << "\n[game " << ep->id << "] step " << ep->pr.steps
              << " action=" << action_name(action)
              << " reward=" << sr.reward
              << " score=" << ep->env.score()
              << " max_tile=" << ep->pr.max_tile << "\n";
        print_board(ep->env.board(), *os);
      }

      const bool done =
        ep->env.is_terminal() || (int)ep->pr.steps >= mcfg->game.max_steps;

      if (!done) {
        // enqueue next step-search
        wq->push(std::make_unique<StepSearchWork>(
          wq, model, be, cfg_base, max_inflight_per_task,
          mcfg, done_games, total_games, ep, results, os_mu, os
        ));
        return WorkOutcome::Finished;
      }

      // finalize
      (*results)[(size_t)ep->id] = ep->pr;

      const int d = done_games->fetch_add(1) + 1;
      if (d >= total_games) wq->stop();

      return WorkOutcome::Finished;
    }
  };

  // Header
  {
    std::lock_guard<std::mutex> lk(os_mu);
    os << "=== 2048 play_many (MCTS) ===\n"
       << "games=" << mcfg.games
       << " sims=" << mcfg.game.sims
       << " cpuct=" << mcfg.game.cpuct
       << " temp=" << mcfg.game.temperature
       << " eval_max_batch=" << mcfg.game.eval_max_batch
       << " eval_max_wait_us=" << mcfg.game.eval_max_wait_us
       << "\n";
    if (mcfg.print_game >= 0) {
      os << "printing game=" << mcfg.print_game
         << " print_each_step=" << (mcfg.game.print_each_step ? "true" : "false")
         << "\n";
    }
  }

  // Initialize episodes (enqueue first step work for each)
  for (int i = 0; i < mcfg.games; ++i) {
    auto ep = std::make_shared<EpisodeCtx>();
    ep->id = i;

    ep->ep_seed = splitmix64(cfg_base.seed ^ (0xA5A5A5A5A5A5A5A5ull + (std::uint64_t)i));
    ep->episode_seed = splitmix64(ep->ep_seed ^ 0xC3C3C3C3C3C3C3C3ull);

    ep->game_rng   = env2048::RNG(splitmix64(ep->ep_seed ^ 0x1111111111111111ull));
    ep->sample_rng = env2048::RNG(splitmix64(ep->ep_seed ^ 0x2222222222222222ull));

    ep->env.reset(ep->game_rng, mcfg.game.initial_tiles);

    ep->pr.final_board = ep->env.board();
    ep->pr.max_tile = ep->env.max_tile();

    if (mcfg.print_game >= 0 && i == mcfg.print_game && mcfg.print_initial_board) {
      std::lock_guard<std::mutex> lk(os_mu);
      os << "\n[game " << i << "] initial board:\n";
      print_board(ep->env.board(), os);
    }

    wq.push(std::make_unique<StepSearchWork>(
      &wq, &amodel, be, cfg_base, max_inflight_per_task,
      &mcfg, &done_games, mcfg.games, ep, &out.results, &os_mu, &os
    ));
  }

  // Worker pool
  const int hw = (int)std::thread::hardware_concurrency();
  const int workers = std::max(1, (mcfg.workers > 0) ? mcfg.workers : (hw > 0 ? hw : 1));

  std::vector<std::thread> pool;
  pool.reserve((size_t)workers);

  for (int t = 0; t < workers; ++t) {
    pool.emplace_back([&]{
      while (true) {
        auto item = wq.pop();
        if (!item) break;

        WorkOutcome r = item->pump(mcfg.sim_budget);
        if (r == WorkOutcome::Finished) continue;

        // requeue
        wq.push(std::move(item));
      }
    });
  }

  for (auto& th : pool) th.join();

  // Aggregates
  double sum_r = 0.0;
  double sum_s = 0.0;
  std::uint32_t best_tile = 0;

  for (const auto& r : out.results) {
    sum_r += (double)r.total_reward;
    sum_s += (double)r.steps;
    best_tile = std::max(best_tile, r.max_tile);
  }

  out.avg_total_reward = sum_r / (double)out.results.size();
  out.avg_steps = sum_s / (double)out.results.size();
  out.best_max_tile = best_tile;

  // -----------------------
  // Max-tile statistics
  // -----------------------
  {
    const double n = (double)out.results.size();

    // exact distribution: max_tile == X
    std::map<std::uint32_t, std::uint32_t> exact_counts;
    for (const auto& r : out.results) exact_counts[r.max_tile] += 1;

    out.max_tile_exact.clear();
    out.max_tile_exact.reserve(exact_counts.size());
    for (const auto& kv : exact_counts) {
      const std::uint32_t tile = kv.first;
      const std::uint32_t cnt  = kv.second;
      out.max_tile_exact.push_back(TileStat{
        tile,
        cnt,
        (n > 0.0 ? 100.0 * (double)cnt / n : 0.0)
      });
    }

    // reach rate: max_tile >= threshold (128, 256, 512, ...)
    const std::uint32_t cap = std::max<std::uint32_t>(2048u, best_tile);
    std::vector<std::uint32_t> thresholds;
    for (std::uint32_t t = 128; t != 0 && t <= cap; t <<= 1) thresholds.push_back(t);
    if (thresholds.empty()) thresholds.push_back(128);

    out.max_tile_at_least.clear();
    out.max_tile_at_least.reserve(thresholds.size());
    for (std::uint32_t t : thresholds) {
      std::uint32_t cnt = 0;
      for (const auto& r : out.results) if (r.max_tile >= t) ++cnt;

      out.max_tile_at_least.push_back(TileStat{
        t,
        cnt,
        (n > 0.0 ? 100.0 * (double)cnt / n : 0.0)
      });
    }
  }

  {
    std::lock_guard<std::mutex> lk(os_mu);

    const auto old_flags = os.flags();
    const auto old_prec  = os.precision();
    os.setf(std::ios::fixed);
    os.precision(2);

    os << "\n=== play_many done ===\n"
       << "avg_total_reward=" << out.avg_total_reward
       << " avg_steps=" << out.avg_steps
       << " best_max_tile=" << out.best_max_tile
       << "\n";

    os << "\nmax_tile distribution (exact):\n";
    for (const auto& s : out.max_tile_exact) {
      os << "  " << std::setw(6) << s.tile
         << " : " << std::setw(6) << s.count
         << "  (" << std::setw(6) << s.pct << "%)\n";
    }

    os << "\nmax_tile reach rate (>= tile):\n";
    for (const auto& s : out.max_tile_at_least) {
      os << "  >= " << std::setw(6) << s.tile
         << " : " << std::setw(6) << s.count
         << "  (" << std::setw(6) << s.pct << "%)\n";
    }

    os.flags(old_flags);
    os.precision(old_prec);
  }

  return out;
}

} // namespace play2048
