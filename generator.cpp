// generator.cpp
#include "generator.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>


// ---------------------------
// Misc helpers
// ---------------------------
double temperature_for_step(std::uint64_t train_step) {
  if (train_step < 100000ULL) return 1.0;
  if (train_step < 200000ULL) return 0.5;
  if (train_step < 300000ULL) return 0.1;
  return 0.0;
}

// ---------------------------
// Scheduler work queue
// ---------------------------
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

// ---------------------------
// Episode context
// ---------------------------
struct EpisodeCtx {
  int ep = 0;
  std::uint64_t ep_seed = 0;
  std::uint64_t episode_seed = 0;

  env2048::RNG game_rng{0};    // for env.reset/env.step only
  env2048::RNG sample_rng{0};  // for action sampling only
  env2048::Env env;

  std::vector<Step> steps;

  explicit EpisodeCtx(int ep_) : ep(ep_) {}
};

// ---------------------------
// Generator
// ---------------------------
Generator::Generator(rl2048::Net net, torch::Device device, const GeneratorConfig& cfg, std::shared_ptr<Logger> logger)
  : net_(std::move(net)), device_(device), cfg_(cfg), logger_(std::move(logger)) {
  net_->to(device_);
  net_->eval();
}

void Generator::generate_into(ReplayBuffer& replay, std::uint64_t train_step) {
  auto logger = logger_;

  net_->eval();

  const double temp_run = temperature_for_step(train_step);

  // One shared evaluator thread + one shared async model
  auto be = std::make_shared<BatchedEvaluator>(
    net_, device_,
    /*max_batch=*/cfg_.eval_max_batch,
    /*max_wait_us=*/cfg_.eval_max_wait_us
  );
  TorchAsyncMCTSModel amodel(be);

  // Base MCTS config (per-step seed is overridden)
  agmcts::MCTSConfig cfg_base;
  cfg_base.num_simulations = cfg_.sims;
  cfg_base.cpuct = cfg_.cpuct;
  cfg_base.dirichlet_alpha = cfg_.dirichlet_alpha;
  cfg_base.dirichlet_epsilon = cfg_.dirichlet_epsilon;
  cfg_base.temperature = temp_run;
  cfg_base.seed = 0xC0FFEEULL ^ cfg_.seed ^ (std::uint64_t)train_step;
  cfg_base.use_afterstate_value = cfg_.afterstate;

  using Backup = DiscountedReturnBackup<S2048, agmcts::NoPayload>;
  using SearchTask = agmcts::mcts_task<S2048, TorchAsyncMCTSModel, agmcts::NoPayload, Backup>;

  // Per-task throttle: allow some inflight leaves per search
  const int max_inflight_per_task = std::max(32, cfg_.eval_max_batch * 4);

  WorkQueue wq;

  std::atomic<int> done_episodes{0};

  // Factory: build a step-search work item for an episode
  struct StepSearchWork final : WorkItem {
    // logger
    std::shared_ptr<Logger> logger = nullptr;

    // refs
    WorkQueue* wq = nullptr;
    ReplayBuffer* replay = nullptr;
    TorchAsyncMCTSModel* model = nullptr;
    std::shared_ptr<BatchedEvaluator> be;

    agmcts::MCTSConfig cfg_base;
    int save_tail = 0;
    double temp_run = 1.0;
    int max_inflight_per_task = 0;
    std::atomic<int>* done_episodes = nullptr;
    int total_episodes = 0;

    std::shared_ptr<EpisodeCtx> ep;

    // search state
    std::unique_ptr<SearchTask> search;

    StepSearchWork(
      WorkQueue* wq_,
      ReplayBuffer* replay_,
      TorchAsyncMCTSModel* model_,
      std::shared_ptr<BatchedEvaluator> be_,
      std::shared_ptr<Logger> logger_,
      const agmcts::MCTSConfig& cfg_base_,
      int save_tail_,
      double temp_run_,
      int max_inflight_per_task_,
      std::atomic<int>* done_episodes_,
      int total_episodes_,
      std::shared_ptr<EpisodeCtx> ep_)
      : wq(wq_), replay(replay_), model(model_), be(std::move(be_)),
        logger(std::move(logger_)), cfg_base(cfg_base_),
        save_tail(save_tail_), temp_run(temp_run_),
        max_inflight_per_task(max_inflight_per_task_),
        done_episodes(done_episodes_), total_episodes(total_episodes_),
        ep(std::move(ep_)) {

      // Build per-step search task
      const auto board = ep->env.board();
      agmcts::MCTSConfig cfg_step = cfg_base;

      cfg_step.seed = splitmix64(cfg_base.seed ^ ep->ep_seed ^ (std::uint64_t)ep->steps.size());

      const std::uint64_t root_seed = splitmix64(ep->episode_seed ^ (uint64_t)board ^ (0xA5A5A5A5A5A5A5A5ull + ep->steps.size()));

      S2048 root(board);
      root.reward_from_parent_raw = 0.0;

      search = std::make_unique<SearchTask>(cfg_step, model, root, max_inflight_per_task);
    }

    WorkOutcome pump(int sim_budget) override {
      auto st = search->pump(sim_budget);

      if (st == SearchTask::PumpStatus::Waiting) {
        // Backoff a bit to avoid busy spin when everything is inflight
        const uint64_t last = be->epoch();
        be->wait_for_epoch_change(last, /*timeout_us=*/200);
        return WorkOutcome::RequeueWaiting;
      }

      if (st != SearchTask::PumpStatus::Done) {
        return WorkOutcome::RequeueProgress;
      }

      // Done: consume search result -> pick action -> step env -> enqueue next work or finalize
      auto res = search->result();

      std::array<float,4> pi = {0,0,0,0};

      // Root should be a PLAYER node => root moves should be Act(0..3)
      if (temp_run <= 1e-12) {
        int best_a = 0, best_v = -1;
        for (const auto& e : res.root_entries) {
          if (e.move.k != S2048::Move::Act) continue; // defensive
          if (e.visits > best_v) { best_v = e.visits; best_a = (int)e.move.v; }
        }
        pi[best_a] = 1.0f;
      } else {
        auto dist = res.visit_distribution(temp_run);
        for (auto& [mv, p] : dist) {
          if (mv.k != S2048::Move::Act) continue;     // defensive
          pi[mv.v] = (float)p;
        }
      }

      env2048::Action action;
      if (temp_run <= 1e-12) {
        int best = 0;
        for (int a = 1; a < 4; ++a) if (pi[a] > pi[best]) best = a;
        action = static_cast<env2048::Action>(best);
      } else {
        action = sample_action(pi, ep->sample_rng);
      }

      // visit-weighted Q at root
      double root_value_raw = 0.0;
      int total_visits = 0;
      for (const auto& e : res.root_entries) {
        if (e.move.k == S2048::Move::Act) total_visits += e.visits;
      }
      if (total_visits > 0) {
        for (const auto& e : res.root_entries) {
          if (e.move.k != S2048::Move::Act) continue;
          root_value_raw += (double(e.visits) / double(total_visits)) * e.q;
        }
      }

      // step env
      const auto board_before = ep->env.board();

      // compute deterministic afterstate (no spawn) on a copy
      env2048::Env env_det = ep->env;
      auto mr = env_det.move(action);
      const std::uint64_t after_board = (std::uint64_t)env_det.board();

      // now do the real environment step (includes spawn)
      auto sr = ep->env.step(action, ep->game_rng);

      Step stp;
      stp.board = (std::uint64_t)board_before;
      stp.after_board = after_board;
      stp.action = (std::uint8_t)action;
      stp.reward = sr.reward;
      stp.pi = pi;
      stp.root_value_raw = (float)root_value_raw;
      ep->steps.push_back(stp);


      if (!ep->env.is_terminal()) {
        // Enqueue next step-search for this episode
        wq->push(std::make_unique<StepSearchWork>(
          wq, replay, model, be, logger,
          cfg_base, save_tail, temp_run, max_inflight_per_task,
          done_episodes, total_episodes, ep
        ));
        return WorkOutcome::Finished;
      }

      // Episode finished -> push tail into replay + log
      const int start = std::max(0, (int)ep->steps.size() - save_tail);
      std::vector<Step> tail(ep->steps.begin() + start, ep->steps.end());
      replay->push_episode(tail);

      std::uint32_t total_reward = 0;
      for (const auto& s : ep->steps) total_reward += s.reward;

      logf(*logger,
           "gen ep=", ep->ep,
           " steps=", ep->steps.size(),
           " total_reward=", total_reward,
           " max_tile=", ep->env.max_tile(),
           " temp=", temp_run,
           " replay_size=", replay->size());

      const int d = done_episodes->fetch_add(1) + 1;
      if (d >= total_episodes) {
        wq->stop();
      }

      return WorkOutcome::Finished;
    }
  };

  // Initialize episodes and enqueue first step-search per episode
  {
    for (int ep = 0; ep < cfg_.episodes; ++ep) {
      auto ctx = std::make_shared<EpisodeCtx>(ep);

      ctx->ep_seed = splitmix64(cfg_base.seed ^ (0xA5A5A5A5A5A5A5A5ull + (std::uint64_t)ep));
      ctx->game_rng   = env2048::RNG(splitmix64(ctx->ep_seed ^ 0x1111111111111111ull));
      ctx->sample_rng = env2048::RNG(splitmix64(ctx->ep_seed ^ 0x2222222222222222ull));

      ctx->env.reset(ctx->game_rng, 2);
      ctx->steps.reserve(256);

      ctx->episode_seed = splitmix64(ctx->ep_seed ^ 0xC3C3C3C3C3C3C3C3ull);

      wq.push(std::make_unique<StepSearchWork>(
        &wq, &replay, &amodel, be, logger,
        cfg_base, cfg_.save_tail, temp_run, max_inflight_per_task,
        &done_episodes, cfg_.episodes, ctx
      ));
    }
  }

  // Worker pool: pumps many step-search tasks round-robin
  const int workers = std::max(1, cfg_.workers);
  std::vector<std::thread> pool;
  pool.reserve((size_t)workers);

  const int sim_budget = 16;

  for (int t = 0; t < workers; ++t) {
    pool.emplace_back([&]{
      while (true) {
        auto item = wq.pop();
        if (!item) break;

        WorkOutcome out = item->pump(sim_budget);

        if (out == WorkOutcome::Finished) {
          continue;
        }

        // requeue
        wq.push(std::move(item));
      }
    });
  }

  for (auto& th : pool) th.join();
}
