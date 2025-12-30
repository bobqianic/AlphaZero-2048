// generator.cpp
#include "generator.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <ATen/cuda/CUDAEvent.h>
#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>


static constexpr double kDiscount = 0.999;

// ---------------------------
// 2048 state for MCTS
// ---------------------------
struct S2048 {
  using Move = env2048::Action;
  using Player = int; // single-agent

  env2048::Env env;
  env2048::RNG rng;

  double reward_from_parent_raw = 0.0;

  S2048() = default;
  explicit S2048(env2048::Env::Board b, std::uint64_t seed) : env(b), rng(seed) {}

  Player current_player() const { return 0; }
  bool is_terminal() const { return env.is_terminal(); }
  double terminal_value(Player /*perspective*/) const { return 0.0; }

  std::vector<Move> legal_moves() const {
    std::vector<Move> m;
    const std::uint8_t mask = env.legal_actions_mask();
    for (int a = 0; a < env2048::kNumActions; ++a) {
      if (mask & (1u << a)) m.push_back(static_cast<Move>(a));
    }
    return m;
  }

  void apply_move(const Move& mv) {
    auto sr = env.step(mv, rng);
    reward_from_parent_raw = static_cast<double>(sr.reward);
  }
};

// ---------------------------
// Backup strategy: discounted return (single-agent).
// ---------------------------
template <class State, class Payload>
struct DiscountedReturnBackup {
  using EvalPayload = Payload;
  using BackupValue = double;

  static BackupValue make_leaf_value(double leaf_value,
                                     const EvalPayload* /*payload*/,
                                     const State& /*leaf_state*/) {
    return leaf_value;
  }

  static BackupValue move_to_parent(const BackupValue& v_child,
                                   const State& /*parent_state*/,
                                   const State& child_state) {
    const double r = child_state.reward_from_parent_raw;
    return r + kDiscount * v_child;
  }

  static void update_node(agmcts::Node<State, DiscountedReturnBackup>& node,
                          const BackupValue& /*v*/) {
    node.N += 1;
  }

  static void update_edge(agmcts::Edge<State, DiscountedReturnBackup>& edge,
                          const BackupValue& v_child) {
    const double r = edge.child ? edge.child->state.reward_from_parent_raw : 0.0;
    const double v_edge = r + kDiscount * v_child;

    edge.N += 1;
    edge.W += v_edge;
    edge.Q = edge.W / std::max(1, edge.N);
  }
};

// ---------------------------
// Async Batched Evaluator (submit + poll)
// ---------------------------
struct EvalOut {
  std::array<double,4> policy;
  double value;
};

struct EvalHandle {
  env2048::Env::Board board{};
  EvalOut out{};
  std::atomic<bool> ready{false};
};

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

class BatchedEvaluator {
public:
  BatchedEvaluator(rl2048::Net net,
                   torch::Device device,
                   int max_batch,
                   int max_wait_us)
      : net_(std::move(net)),
        device_(device),
        max_batch_(std::max(1, max_batch)),
        max_wait_us_(std::max(1, max_wait_us)) {

    net_->to(device_);
    net_->eval();

    support_ = torch::arange(
                 0, rl2048::kSupportMax + 1,
                 torch::TensorOptions().dtype(torch::kFloat32).device(device_))
                 .view({1, -1});

    init_slots();
    worker_ = std::thread([this] { loop(); });
  }

  ~BatchedEvaluator() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  std::shared_ptr<EvalHandle> submit(env2048::Env::Board b) {
    auto h = std::make_shared<EvalHandle>();
    h->board = b;
    {
      std::lock_guard<std::mutex> lk(mu_);
      q_.push_back(h);
    }
    cv_.notify_one();
    return h;
  }

  uint64_t epoch() const { return epoch_.load(std::memory_order_relaxed); }

  void wait_for_epoch_change(uint64_t last_epoch, int timeout_us) {
    std::unique_lock<std::mutex> lk(epoch_mu_);
    epoch_cv_.wait_for(
      lk,
      std::chrono::microseconds(timeout_us),
      [&] { return epoch_.load(std::memory_order_relaxed) != last_epoch; }
    );
  }

private:
  rl2048::Net net_;
  torch::Device device_;
  int max_batch_;
  int max_wait_us_;

  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<EvalHandle>> q_;
  bool stop_ = false;
  std::thread worker_;

  torch::Tensor support_;

  std::atomic<uint64_t> epoch_{0};
  std::mutex epoch_mu_;
  std::condition_variable epoch_cv_;

  struct Slot {
    torch::Tensor boards_pinned; // [maxB] pinned CPU int64
    torch::Tensor boards_dev;    // [maxB] GPU int64

    torch::Tensor xb_dev;        // [maxB, obs] GPU float32
    torch::Tensor p_pinned;      // [maxB, 4] pinned CPU float
    torch::Tensor v_pinned;      // [maxB] pinned CPU float

    std::optional<at::cuda::CUDAStream> stream;
    at::cuda::CUDAEvent done;

    bool in_use = false;
    int B = 0;
    std::vector<std::shared_ptr<EvalHandle>> batch;
  };

  static constexpr int kSlots = 8;
  std::array<Slot, kSlots> slots_{};

  void init_slots() {
    auto pinned_f32 = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(torch::kCPU)
                    .pinned_memory(true);

    auto pinned_i64 = torch::TensorOptions()
                        .dtype(torch::kInt64)
                        .device(torch::kCPU)
                        .pinned_memory(true);

    auto dev_i64 = torch::TensorOptions()
                     .dtype(torch::kInt64)
                     .device(device_);

    auto dev_f32 = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .device(device_);

    for (auto& s : slots_) {
      s.boards_pinned = torch::empty({max_batch_}, pinned_i64);
      s.boards_dev    = torch::empty({max_batch_}, dev_i64);

      s.xb_dev        = torch::empty({max_batch_, (int64_t)rl2048::kObsDim}, dev_f32);

      s.p_pinned      = torch::empty({max_batch_, 4}, pinned_f32);
      s.v_pinned      = torch::empty({max_batch_}, pinned_f32);

      s.stream = at::cuda::getStreamFromPool(false, device_.index());
    }
  }

  std::vector<std::shared_ptr<EvalHandle>> take_batch_from_queue() {
    using clock = std::chrono::steady_clock;

    std::vector<std::shared_ptr<EvalHandle>> batch;
    batch.reserve((size_t)max_batch_);

    auto drain = [&] {
      while (!q_.empty() && (int)batch.size() < max_batch_) {
        batch.emplace_back(std::move(q_.front()));
        q_.pop_front();
      }
    };

    auto any_in_flight = [&] {
      for (auto& s : slots_) if (s.in_use) return true;
      return false;
    };

    std::unique_lock<std::mutex> lk(mu_);

    if (q_.empty() && !stop_) {
      if (any_in_flight()) {
        cv_.wait_for(lk, std::chrono::microseconds(200),
                     [&] { return stop_ || !q_.empty(); });
      } else {
        cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
      }
    }

    if (stop_ && q_.empty()) return batch;
    if (q_.empty()) return batch;

    drain();

    const auto deadline = clock::now() + std::chrono::microseconds(max_wait_us_);

    while (!stop_ && (int)batch.size() < max_batch_) {
      if (clock::now() >= deadline) break;

      if (q_.empty()) {
        if (!cv_.wait_until(lk, deadline, [&] { return stop_ || !q_.empty(); })) break;
      }
      drain();
    }

    return batch;
  }

  void loop() {
    torch::InferenceMode im;
    at::cuda::CUDAGuard device_guard(device_.index());

    while (true) {
      // 1) Retire completed slots (non-blocking)
      for (auto& s : slots_) {
        if (!s.in_use || !s.done.query()) continue;

        auto p_view = s.p_pinned.narrow(0, 0, s.B);
        auto v_view = s.v_pinned.narrow(0, 0, s.B);

        const float* pptr = p_view.data_ptr<float>();
        const float* vptr = v_view.data_ptr<float>();

        for (int i = 0; i < s.B; ++i) {
          auto& h = s.batch[i];
          for (int a = 0; a < 4; ++a) h->out.policy[a] = pptr[i * 4 + a];
          h->out.value = (double)vptr[i];
          h->ready.store(true, std::memory_order_release);
        }

        s.batch.clear();
        s.in_use = false;

        epoch_.fetch_add(1, std::memory_order_relaxed);
        epoch_cv_.notify_all();
      }

      // 2) Find a free slot
      Slot* slot = nullptr;
      for (auto& s : slots_) {
        if (!s.in_use) { slot = &s; break; }
      }

      // (1) If no slot is free, DO NOT sleep — block until some in-flight batch finishes.
      if (!slot) {
        for (auto& s : slots_) {
          if (s.in_use) { s.done.synchronize(); break; }
        }
        continue; // next loop iteration will retire immediately
      }

      // 3) Build batch from queue
      auto batch = take_batch_from_queue();
      if (batch.empty()) {
        if (stop_) return;
        continue;
      }

      slot->B = (int)batch.size();
      slot->batch = std::move(batch);
      slot->in_use = true;

      auto b_cpu = slot->boards_pinned.narrow(0, 0, slot->B);
      int64_t* bptr = b_cpu.data_ptr<int64_t>();
      for (int i = 0; i < slot->B; ++i) {
        bptr[i] = (int64_t)slot->batch[i]->board; // preserve bit pattern
      }

      // 4) Launch GPU work on THIS SLOT'S stream (2), no sync
      {
        // Guard current stream to this slot's stream
        c10::cuda::CUDAStreamGuard sg(slot->stream.value().unwrap());

        auto b_dev  = slot->boards_dev.narrow(0, 0, slot->B);
        auto xb_dev = slot->xb_dev.narrow(0, 0, slot->B);

        // async H2D of just 8 bytes/board
        b_dev.copy_(b_cpu, /*non_blocking=*/true);

        // GPU expand boards -> xb_dev
        encode2048_onehot31_out_cuda(b_dev, xb_dev);

        torch::Tensor pl, vl, rl;
        {
          AutocastGuard amp(at::kCUDA, at::kBFloat16, true);
          std::tie(pl, vl, rl) = net_->forward(xb_dev);
        }

        auto p_dev = torch::softmax(pl.to(torch::kFloat32), 1);
        auto v_dev = rl2048::decode_value_raw(vl.to(torch::kFloat32), support_, rl2048::kEpsTransform);

        auto p_cpu = slot->p_pinned.narrow(0, 0, slot->B);
        auto v_cpu = slot->v_pinned.narrow(0, 0, slot->B);

        p_cpu.copy_(p_dev, /*non_blocking=*/true);
        v_cpu.copy_(v_dev, /*non_blocking=*/true);

        // Record completion on the same stream explicitly
        slot->done.record();
      }
    }
  }
};


// ---------------------------
// Async model wrapper for mcts_task
// ---------------------------
struct TorchAsyncMCTSModel {
  using Ticket = std::shared_ptr<EvalHandle>;
  std::shared_ptr<BatchedEvaluator> be;

  explicit TorchAsyncMCTSModel(std::shared_ptr<BatchedEvaluator> eval) : be(std::move(eval)) {
    if (!be) throw std::runtime_error("TorchAsyncMCTSModel: BatchedEvaluator is null");
  }

  Ticket submit(const S2048& s) {
    return be->submit(s.env.board());
  }

  bool poll(const Ticket& t, agmcts::Evaluation<env2048::Action, agmcts::NoPayload>* out) {
    if (!t->ready.load(std::memory_order_acquire)) return false;

    const EvalOut& o = t->out;
    out->value = o.value;
    out->policy = {
      {env2048::Action::Up,    o.policy[0]},
      {env2048::Action::Right, o.policy[1]},
      {env2048::Action::Down,  o.policy[2]},
      {env2048::Action::Left,  o.policy[3]},
    };
    return true;
  }
};

// ---------------------------
// Misc helpers
// ---------------------------
double temperature_for_step(std::uint64_t train_step) {
  if (train_step < 100000ULL) return 1.0;
  if (train_step < 200000ULL) return 0.5;
  if (train_step < 300000ULL) return 0.1;
  return 0.0;
}

static inline std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static env2048::Action sample_action(const std::array<float,4>& probs, env2048::RNG& rng) {
  const double u = double(rng.next_u32_raw()) / 4294967296.0;
  double c = 0.0;
  for (int a = 0; a < 4; ++a) {
    c += probs[a];
    if (u <= c) return static_cast<env2048::Action>(a);
  }
  int best = 0;
  for (int a = 1; a < 4; ++a) if (probs[a] > probs[best]) best = a;
  return static_cast<env2048::Action>(best);
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

      S2048 root(board, root_seed);
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

      if (temp_run <= 1e-12) {
        int best_a = 0, best_v = -1;
        for (const auto& e : res.root_entries) {
          if (e.visits > best_v) { best_v = e.visits; best_a = (int)e.move; }
        }
        pi[best_a] = 1.0f;
      } else {
        auto dist = res.visit_distribution(temp_run);
        for (auto& [mv, p] : dist) pi[(int)mv] = (float)p;
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
      for (const auto& e : res.root_entries) total_visits += e.visits;
      if (total_visits > 0) {
        for (const auto& e : res.root_entries) {
          root_value_raw += (double(e.visits) / double(total_visits)) * e.q;
        }
      }

      // step env
      const auto board_before = ep->env.board();
      auto sr = ep->env.step(action, ep->game_rng);

      Step stp;
      stp.board = (std::uint64_t)board_before;
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

  const int sim_budget = 16; // tune: 8..32 typical

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
