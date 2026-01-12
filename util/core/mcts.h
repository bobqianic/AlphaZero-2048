///
// Created by qianp on 27/12/2025.
//

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace agmcts {

// ---------------------------
// MinMaxStats (MuZero-style)
// ---------------------------
struct MinMaxStats {
  double minimum =  std::numeric_limits<double>::infinity();
  double maximum = -std::numeric_limits<double>::infinity();

  void reset() {
    minimum =  std::numeric_limits<double>::infinity();
    maximum = -std::numeric_limits<double>::infinity();
  }

  void update(double value) {
    if (!std::isfinite(value)) return;
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
  }

  double normalize(double value) const {
    if (!std::isfinite(value)) return 0.5;
    if (maximum > minimum + 1e-12) {
      return (value - minimum) / (maximum - minimum);
    }
    return 0.5;
  }
};

// ---------------------------
// Configuration
// ---------------------------
struct MCTSConfig {
  int num_simulations = 800;
  double cpuct = 1.5;

  double dirichlet_alpha = 0.3;
  double dirichlet_epsilon = 0.25;

  double temperature = 1.0;

  uint64_t seed = 0xC0FFEEULL;

  // if true, evaluate value on afterstates (chance nodes).
  bool use_afterstate_value = false;
};

// ---------------------------
// Evaluation result (Model -> MCTS)
// ---------------------------
template <class Move, class Payload>
struct Evaluation {
  std::vector<std::pair<Move, double>> policy;
  double value = 0.0;
  Payload payload{};
};

using NoPayload = struct {};

// ---------------------------
// Tree node/edge
// ---------------------------
template <class State, class BackupStrategy>
struct Node;

template <class State, class BackupStrategy>
struct Edge {
  using Move = typename State::Move;

  Move move{};
  double prior = 0.0;

  int N = 0;
  double W = 0.0;
  double Q = 0.0;

  std::unique_ptr<Node<State, BackupStrategy>> child;
};

template <class State, class BackupStrategy>
struct Node {
  using Player = typename State::Player;
  using Move = typename State::Move;

  State state;
  Player to_play{};

  bool expanded = false;

  // For async mcts_task: leaf eval is submitted and pending.
  bool inflight = false;

  int N = 0;

  std::vector<Edge<State, BackupStrategy>> edges;

  explicit Node(const State& s) : state(s), to_play(s.current_player()) {}
};

// ---------------------------
// BackupStrategy concept + default
// ---------------------------
template <class State, class Payload>
struct DefaultBackupStrategy {
  using EvalPayload = Payload;
  using BackupValue = double;

  static BackupValue make_leaf_value(double leaf_value,
                                     const EvalPayload* /*payload*/,
                                     const State& /*leaf_state*/) {
    return leaf_value;
  }

  static BackupValue move_to_parent(const BackupValue& v,
                                    const State& /*parent_state*/,
                                    const State& /*child_state*/) {
    return -v;
  }

  static void update_node(Node<State, DefaultBackupStrategy>& node,
                          const BackupValue& /*v*/) {
    node.N += 1;
  }

  static void update_edge(Edge<State, DefaultBackupStrategy>& edge,
                          const BackupValue& v) {
    edge.N += 1;
    edge.W += v;
    edge.Q = edge.W / std::max(1, edge.N);
  }
};

// ---------------------------
// Helpers: dirichlet noise
// ---------------------------
inline std::vector<double> sample_dirichlet(std::mt19937_64& rng, int k, double alpha) {
  std::gamma_distribution<double> gamma(alpha, 1.0);
  std::vector<double> x(k, 0.0);
  double sum = 0.0;
  for (int i = 0; i < k; ++i) {
    x[i] = std::max(0.0, gamma(rng));
    sum += x[i];
  }
  if (sum <= 0.0) {
    std::fill(x.begin(), x.end(), 1.0 / std::max(1, k));
    return x;
  }
  for (double& v : x) v /= sum;
  return x;
}

// ---------------------------
// Chance-node detection (optional State API)
// ---------------------------
namespace detail {
template <typename T, typename = void>
struct has_is_chance_node : std::false_type {};

template <typename T>
struct has_is_chance_node<T, std::void_t<decltype(std::declval<const T&>().is_chance_node())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_chance_priors : std::false_type {};

template <typename T>
struct has_chance_priors<T, std::void_t<decltype(std::declval<const T&>().chance_priors())>>
    : std::true_type {};

template <typename State>
inline bool is_chance_node(const State& s) {
  if constexpr (has_is_chance_node<State>::value) {
    return (bool)s.is_chance_node();
  } else {
    return false;
  }
}

template <typename State>
inline auto get_chance_priors(const State& s) {
  static_assert(has_chance_priors<State>::value,
                "State has is_chance_node() but does not provide chance_priors()");
  return s.chance_priors();
}

} // namespace detail

// ---------------------------
// MCTS result at root
// ---------------------------
template <class State>
struct RootPolicyEntry {
  typename State::Move move;
  int visits = 0;
  double prior = 0.0;
  double q = 0.0;
};

template <class State>
struct SearchResult {
  std::vector<RootPolicyEntry<State>> root_entries;

  std::vector<std::pair<typename State::Move, double>>
  visit_distribution(double temperature) const {
    std::vector<std::pair<typename State::Move, double>> out;
    if (root_entries.empty()) return out;

    if (temperature <= 1e-12) {
      auto it = std::max_element(root_entries.begin(), root_entries.end(),
                                 [](auto& a, auto& b) { return a.visits < b.visits; });
      out.push_back({it->move, 1.0});
      return out;
    }

    std::vector<double> w(root_entries.size(), 0.0);
    double sum = 0.0;
    const double invT = 1.0 / temperature;
    for (size_t i = 0; i < root_entries.size(); ++i) {
      w[i] = std::pow(std::max(0, root_entries[i].visits), invT);
      sum += w[i];
    }
    if (sum <= 0.0) {
      double u = 1.0 / root_entries.size();
      for (auto& e : root_entries) out.push_back({e.move, u});
      return out;
    }
    for (size_t i = 0; i < root_entries.size(); ++i) {
      out.push_back({root_entries[i].move, w[i] / sum});
    }
    return out;
  }
};

// ---------------------------
// Async/pumpable MCTS task (GPU-friendly)
// ---------------------------
//
// Model concept (minimum requirements):
//   using Ticket = ...;
//   Ticket submit(const State&);
//   bool poll(const Ticket&, EvalT* out_eval);
//
// Optional (for GPU batching, recommended):
//   void drive_batch() / run_batch() / flush() / process()
//   (any one of those names; mcts_task will call it from helper drivers below)
//
template <class State,
          class Model,
          class Payload = NoPayload,
          class BackupStrategy = DefaultBackupStrategy<State, Payload>>
class mcts_task {
public:
  using Move = typename State::Move;
  using EvalT = Evaluation<Move, Payload>;
  using BackupValue = typename BackupStrategy::BackupValue;
  using NodeT = Node<State, BackupStrategy>;
  using EdgeT = Edge<State, BackupStrategy>;
  using Ticket = typename Model::Ticket;

  enum class PumpStatus { Progress, Waiting, Done };

  mcts_task(MCTSConfig cfg, Model* model, const State& root_state, int max_inflight = 0)
    : cfg_(cfg), model_(model), rng_(cfg.seed), max_inflight_(max_inflight) {
    assert(model_ && "Model pointer must not be null");
    root_ = std::make_unique<NodeT>(root_state);
    min_max_.reset();

    if (root_->state.is_terminal()) {
      completed_sims_ = cfg_.num_simulations;
    }
  }

  bool done() const { return completed_sims_ >= cfg_.num_simulations; }

  SearchResult<State> result() const {
    return collect_root_result(*root_);
  }

  PumpStatus pump(int sim_budget) {
    if (done()) return PumpStatus::Done;

    bool did_any = false;

    // 0) Ensure root is expanded/submitted
    did_any |= ensure_root_ready();

    // 1) Reap completed evals
    did_any |= reap_ready();

    if (!root_->expanded && !root_->state.is_terminal()) {
      return did_any ? PumpStatus::Progress : PumpStatus::Waiting;
    }

    // 2) Run bounded number of simulations without blocking
    for (int i = 0; i < sim_budget && !done(); ++i) {
      auto st = simulate_once();
      if (st == SimStepStatus::Completed || st == SimStepStatus::Submitted) {
        did_any = true;
        continue;
      }
      break;
    }

    // 3) Reap again
    did_any |= reap_ready();

    if (done()) return PumpStatus::Done;
    return did_any ? PumpStatus::Progress : PumpStatus::Waiting;
  }

  // Expose root if you want to reuse/inspect tree.
  const NodeT* root() const { return root_.get(); }

private:
  struct PathStep {
    NodeT* node = nullptr;
    EdgeT* edge = nullptr;
  };

  enum class PendingKind { PlayerPolicyExpand, ChanceValueEval };

  struct Pending {
    NodeT* leaf = nullptr;
    std::vector<PathStep> path;
    Ticket ticket;
    PendingKind kind;
  };

  enum class SimStepStatus { Completed, Submitted, Waiting };

  MCTSConfig cfg_;
  Model* model_;
  std::mt19937_64 rng_;
  std::unique_ptr<NodeT> root_;
  MinMaxStats min_max_;

  int completed_sims_ = 0;
  int max_inflight_ = 0; // 0 => unlimited
  std::deque<Pending> pending_;

  bool root_noise_applied_ = false;
  bool root_eval_submitted_ = false;
  Ticket root_ticket_{};

  // Chance sampling with inflight skip (avoid repeatedly picking blocked edges)
  EdgeT* select_chance_sample_skip_inflight(NodeT& node) {
    double sum = 0.0;
    for (auto& e : node.edges) {
      if (e.child && e.child->inflight && !e.child->expanded && !e.child->state.is_terminal()) continue;
      sum += std::max(0.0, e.prior);
    }
    if (sum <= 0.0) return nullptr;

    std::uniform_real_distribution<double> dist(0.0, sum);
    double u = dist(rng_);
    double c = 0.0;
    for (auto& e : node.edges) {
      if (e.child && e.child->inflight && !e.child->expanded && !e.child->state.is_terminal()) continue;
      c += std::max(0.0, e.prior);
      if (u <= c) return &e;
    }
    // Fallback: return last non-skipped edge
    for (auto it = node.edges.rbegin(); it != node.edges.rend(); ++it) {
      auto& e = *it;
      if (e.child && e.child->inflight && !e.child->expanded && !e.child->state.is_terminal()) continue;
      return &e;
    }
    return nullptr;
  }

  // selection that skips edges whose child is an inflight, unexpanded leaf
  EdgeT* select_puct_skip_inflight(NodeT& node) {
    const double sqrtN = std::sqrt(std::max(1, node.N));
    EdgeT* best = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();

    for (auto& e : node.edges) {
      if (e.child && e.child->inflight && !e.child->expanded && !e.child->state.is_terminal()) {
        continue;
      }
      const double q = (e.N > 0) ? min_max_.normalize(e.Q) : 0.5;
      const double u = cfg_.cpuct * e.prior * (sqrtN / (1.0 + e.N));
      const double score = q + u;
      if (score > best_score) { best_score = score; best = &e; }
    }
    return best;
  }

  bool ensure_root_ready() {
    if (root_->state.is_terminal()) return false;
    if (root_->expanded) return false;

    // Chance root: expand immediately (no model).
    if (detail::is_chance_node(root_->state)) {
      expand_chance_node(*root_);
      return true;
    }

    if (root_eval_submitted_) return false;

    root_ticket_ = model_->submit(root_->state);
    root_->inflight = true;
    root_eval_submitted_ = true;
    return true;
  }


  SimStepStatus simulate_once() {
    if (max_inflight_ > 0 && (int)pending_.size() >= max_inflight_) {
      return SimStepStatus::Waiting;
    }

    std::vector<PathStep> path;
    path.reserve(256);

    NodeT* node = root_.get();

    while (true) {
      path.push_back({node, nullptr});

      if (node->state.is_terminal()) {
        BackupValue bv = BackupStrategy::make_leaf_value(
          node->state.terminal_value(node->state.current_player()),
          nullptr, node->state);
        backup(path, bv);
        completed_sims_ += 1;
        return SimStepStatus::Completed;
      }

      if (!node->expanded) {

        // ---- Chance node ----
        if (detail::is_chance_node(node->state)) {

          if (cfg_.use_afterstate_value) {
            // Afterstate mode: evaluate VALUE at chance node, backup later.
            if (node->inflight) return SimStepStatus::Waiting;

            Ticket t = model_->submit(node->state);
            node->inflight = true;

            pending_.push_back(Pending{node, std::move(path), std::move(t),
                                       PendingKind::ChanceValueEval});
            return SimStepStatus::Submitted;
          }

          // Normal mode: expand chance immediately and continue.
          expand_chance_node(*node);
          // fallthrough to selection below
        }

        // ---- Player node ----
        else {
          if (cfg_.use_afterstate_value) {
            // Afterstate mode: expand player node with POLICY priors (no backup here).
            if (node->inflight) return SimStepStatus::Waiting;

            Ticket t = model_->submit(node->state);
            node->inflight = true;

            pending_.push_back(Pending{node, {}, std::move(t),
                                       PendingKind::PlayerPolicyExpand});
            return SimStepStatus::Submitted;
          }

          // Normal mode: policy+value leaf at player node (current behavior)
          if (node->inflight) return SimStepStatus::Waiting;

          Ticket t = model_->submit(node->state);
          node->inflight = true;

          pending_.push_back(Pending{node, std::move(path), std::move(t),
                                     PendingKind::ChanceValueEval /*reused below*/});
          // ^ In normal mode you want “leaf eval backup”. Keep your original kind if you prefer.
          return SimStepStatus::Submitted;
        }
      }


      EdgeT* best = nullptr;
      if (detail::is_chance_node(node->state)) {
        best = select_chance_sample_skip_inflight(*node);
      } else {
        best = select_puct_skip_inflight(*node);
      }
      if (!best) return SimStepStatus::Waiting;

      if (!best->child) {
        State child_state = node->state;
        child_state.apply_move(best->move);
        best->child = std::make_unique<NodeT>(child_state);
      }

      path.back().edge = best;
      node = best->child.get();
    }
  }

  bool reap_ready() {
    bool did = false;

    // Root expansion (player root only)
    if (root_eval_submitted_ && !root_->expanded) {
      EvalT eval;
      if (model_->poll(root_ticket_, &eval)) {
        root_->inflight = false;
        expand_with_eval(*root_, eval);

        // Root Dirichlet noise only for player root
        if (!root_noise_applied_ &&
            cfg_.dirichlet_epsilon > 0.0 && !root_->edges.empty() &&
            !detail::is_chance_node(root_->state)) {
          auto noise = sample_dirichlet(rng_, (int)root_->edges.size(), cfg_.dirichlet_alpha);
          for (size_t i = 0; i < root_->edges.size(); ++i) {
            double p = root_->edges[i].prior;
            root_->edges[i].prior =
              (1.0 - cfg_.dirichlet_epsilon) * p + cfg_.dirichlet_epsilon * noise[i];
          }
          renormalize_priors(*root_);
        }
        root_noise_applied_ = true;

        did = true;
      }
    }

    // Reap pending leaf evals
    const int n = (int)pending_.size();
    for (int i = 0; i < n; ++i) {
      Pending p = std::move(pending_.front());
      pending_.pop_front();

      EvalT eval;
      if (!model_->poll(p.ticket, &eval)) {
        pending_.push_back(std::move(p));
        continue;
      }

      p.leaf->inflight = false;

      if (p.kind == PendingKind::PlayerPolicyExpand) {
        // expand player node with network policy priors
        expand_with_eval(*p.leaf, eval);
        did = true;
        continue;
      }

      // ChanceValueEval (afterstate): expand chance priors, then backup using eval.value
      if (detail::is_chance_node(p.leaf->state)) {
        expand_chance_node(*p.leaf);
      } else {
        // normal-mode leaf: expand from eval policy (your original behavior)
        expand_with_eval(*p.leaf, eval);
      }

      BackupValue bv = BackupStrategy::make_leaf_value(eval.value, &eval.payload, p.leaf->state);
      backup(p.path, bv);
      completed_sims_ += 1;
      did = true;
    }

    return did;
  }

  void expand_chance_node(NodeT& node) {
    if (node.expanded) return;

    auto pri = detail::get_chance_priors(node.state);
    if (pri.empty()) {
      node.expanded = true;
      return;
    }

    double sum = 0.0;
    for (auto& kv : pri) sum += std::max(0.0, kv.second);
    if (sum <= 0.0) {
      double u = 1.0 / pri.size();
      for (auto& kv : pri) kv.second = u;
    } else {
      for (auto& kv : pri) kv.second = std::max(0.0, kv.second) / sum;
    }

    node.edges.clear();
    node.edges.reserve(pri.size());
    for (auto& [m, p] : pri) {
      EdgeT e;
      e.move = m;
      e.prior = p;
      node.edges.push_back(std::move(e));
    }
    node.expanded = true;
  }

  void expand_with_eval(NodeT& node, const EvalT& eval) {
    if (node.expanded) return;

    auto legal = node.state.legal_moves();
    if (legal.empty()) {
      node.expanded = true;
      return;
    }

    const double eps = 1e-8;
    std::vector<double> priors(legal.size(), eps);

    for (const auto& [m, p] : eval.policy) {
      for (size_t i = 0; i < legal.size(); ++i) {
        if (legal[i] == m) {
          priors[i] = std::max(eps, p);
          break;
        }
      }
    }

    double sum = std::accumulate(priors.begin(), priors.end(), 0.0);
    if (sum <= 0.0) {
      double u = 1.0 / legal.size();
      std::fill(priors.begin(), priors.end(), u);
    } else {
      for (double& p : priors) p /= sum;
    }

    node.edges.clear();
    node.edges.reserve(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) {
      EdgeT e;
      e.move = legal[i];
      e.prior = priors[i];
      node.edges.push_back(std::move(e));
    }
    node.expanded = true;
  }

  void renormalize_priors(NodeT& node) {
    double sum = 0.0;
    for (auto& e : node.edges) sum += e.prior;
    if (sum <= 0.0) return;
    for (auto& e : node.edges) e.prior /= sum;
  }

  void backup(std::vector<PathStep>& path, BackupValue v) {
    for (int i = (int)path.size() - 1; i >= 0; --i) {
      NodeT& node = *path[i].node;
      BackupStrategy::update_node(node, v);

      if (path[i].edge) {
        BackupStrategy::update_edge(*path[i].edge, v);
        min_max_.update(path[i].edge->Q);

        const State& parent_state = node.state;
        const State& child_state = path[i].edge->child->state;
        v = BackupStrategy::move_to_parent(v, parent_state, child_state);
      }
    }
  }

  SearchResult<State> collect_root_result(const NodeT& root) const {
    SearchResult<State> r;
    r.root_entries.reserve(root.edges.size());
    for (const auto& e : root.edges) {
      RootPolicyEntry<State> ent;
      ent.move = e.move;
      ent.visits = e.N;
      ent.prior = e.prior;
      ent.q = (e.N > 0) ? e.Q : 0.0;
      r.root_entries.push_back(ent);
    }
    std::sort(r.root_entries.begin(), r.root_entries.end(),
              [](auto& a, auto& b) { return a.visits > b.visits; });
    return r;
  }

  void expand_uniform_policy(NodeT& node) {
    if (node.expanded) return;

    auto legal = node.state.legal_moves();
    if (legal.empty()) {
      node.expanded = true;
      return;
    }

    const double u = 1.0 / (double)legal.size();

    // Tiny jitter to break ties deterministically per-seed
    std::uniform_real_distribution<double> jitter(0.0, 1e-6);

    node.edges.clear();
    node.edges.reserve(legal.size());
    for (auto& m : legal) {
      EdgeT e;
      e.move = m;
      e.prior = u + jitter(rng_);
      node.edges.push_back(std::move(e));
    }
    renormalize_priors(node);
    node.expanded = true;
  }
};

} // namespace agmcts
