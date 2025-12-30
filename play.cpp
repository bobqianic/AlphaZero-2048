//
// Created by qianp on 29/12/2025.
//

// play.cpp
//
// Minimal “play a game” driver built to mirror generator.cpp’s MCTS+batched-eval flow,
// but for a single episode with board printing.
//
// ----- API you can call from main.cpp (forward-declare or put in a play.h) -----
//
// namespace play2048 {
//   struct PlayConfig;
//   struct PlayResult;
//   void print_board(env2048::Env::Board b, std::ostream& os);
//   PlayResult play(rl2048::Net net, torch::Device device, const PlayConfig& cfg, std::ostream& os);
// }
//
// ------------------------------------------------------------------------------

#include "play.h"
#include "util/core/env2048.h"
#include "util/core/mcts.h"
#include "model.h"
#include "util/logger.h"
#include "util/cuda/encode2048.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <ATen/cuda/CUDAEvent.h>
#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace play2048 {

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
// Async Batched Evaluator (submit + poll) — same idea as generator.cpp
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

  static constexpr int kSlots = 4; // play-mode: fewer slots is fine
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
      // 1) Retire completed slots
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

      if (!slot) {
        for (auto& s : slots_) {
          if (s.in_use) { s.done.synchronize(); break; }
        }
        continue;
      }

      // 3) Build batch
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

      // 4) Launch GPU work on this slot’s stream
      {
        c10::cuda::CUDAStreamGuard sg(slot->stream.value().unwrap());

        auto b_dev  = slot->boards_dev.narrow(0, 0, slot->B);
        auto xb_dev = slot->xb_dev.narrow(0, 0, slot->B);

        b_dev.copy_(b_cpu, /*non_blocking=*/true);

        // provided elsewhere in your project
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
// Helpers (seeds, sampling, printing)
// ---------------------------
static inline std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static const char* action_name(env2048::Action a) {
  switch (a) {
    case env2048::Action::Up:    return "Up";
    case env2048::Action::Right: return "Right";
    case env2048::Action::Down:  return "Down";
    case env2048::Action::Left:  return "Left";
    default: return "?";
  }
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

  using Backup = DiscountedReturnBackup<S2048, agmcts::NoPayload>;
  using SearchTask = agmcts::mcts_task<S2048, TorchAsyncMCTSModel, agmcts::NoPayload, Backup>;

  const auto board = env.board();
  const std::uint64_t root_seed =
      splitmix64(episode_seed ^ (std::uint64_t)board ^ (0xA5A5A5A5A5A5A5A5ull + step_index));

  S2048 root(board, root_seed);
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
      if (e.visits > best_v) { best_v = e.visits; best_a = (int)e.move; }
    }
    pi[best_a] = 1.0f;
  } else {
    auto dist = res.visit_distribution(pcfg.temperature);
    for (auto& [mv, p] : dist) pi[(int)mv] = (float)p;
  }

  if (pcfg.temperature <= 1e-12) {
    int best = 0;
    for (int a = 1; a < 4; ++a) if (pi[a] > pi[best]) best = a;
    return static_cast<env2048::Action>(best);
  }
  return sample_action(pi, sampling_rng);
}

// ---------------------------
// Main play() API
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

    const auto before = env.board();
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
      // If you want to see packed board values too:
      // os << "before=0x" << std::hex << before << " after=0x" << env.board() << std::dec << "\n";
      (void)before;
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

} // namespace play2048
