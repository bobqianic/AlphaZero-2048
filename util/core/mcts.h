///
// Created by qianp on 27/12/2025.
//

// mcts.h - header-only AlphaGo-style MCTS (C++17)
//
// Features:
// - PUCT selection (AlphaGo Zero style)
// - Pluggable State / Model interfaces
// - Pluggable BackupStrategy to control "what is backpropagated" and how stats are updated
// - Root Dirichlet noise (optional)
// - Temperature sampling from visit counts (optional)
//
// Assumptions (default strategy):
// - 2-player, zero-sum, alternating turns
// - Model value is from the perspective of the player to move at the evaluated state
//
// You can relax/change those assumptions by providing your own BackupStrategy.
//
// ADDITIONS (for async batching):
// - Node has `inflight` flag used by mcts_task (async/pumpable search)
// - New mcts_task class: non-blocking, yields when leaf eval is pending, resumes when ready
//
// Existing `mcts::search()` is unchanged.

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
#include <utility>
#include <vector>

namespace agmcts {

// ---------------------------
// MinMaxStats (MuZero-style)
// Keeps value ranges stable by normalizing Q into [0,1] during selection.
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

  // Root Dirichlet noise (AlphaGo Zero):
  // Set dirichlet_epsilon = 0 to disable.
  double dirichlet_alpha = 0.3;
  double dirichlet_epsilon = 0.25;

  // Temperature for converting visit counts into a policy.
  double temperature = 1.0;

  uint64_t seed = 0xC0FFEEULL;
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

// Default: no payload
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
// Original synchronous MCTS (unchanged)
// ---------------------------
template <class State,
          class Model,
          class Payload = NoPayload,
          class BackupStrategy = DefaultBackupStrategy<State, Payload>>
class mcts {
public:
  using Move = typename State::Move;
  using Player = typename State::Player;
  using EvalT = Evaluation<Move, Payload>;
  using BackupValue = typename BackupStrategy::BackupValue;

  mcts(MCTSConfig cfg, Model* model)
      : cfg_(cfg), model_(model), rng_(cfg.seed) {
    assert(model_ && "Model pointer must not be null");
  }

  SearchResult<State> search(const State& root_state) {
    root_ = std::make_unique<NodeT>(root_state);
    min_max_.reset();

    expand_if_needed(*root_);

    if (cfg_.dirichlet_epsilon > 0.0 && !root_->edges.empty()) {
      auto noise = sample_dirichlet(rng_, (int)root_->edges.size(), cfg_.dirichlet_alpha);
      for (size_t i = 0; i < root_->edges.size(); ++i) {
        double p = root_->edges[i].prior;
        root_->edges[i].prior =
            (1.0 - cfg_.dirichlet_epsilon) * p + cfg_.dirichlet_epsilon * noise[i];
      }
      renormalize_priors(*root_);
    }

    for (int i = 0; i < cfg_.num_simulations; ++i) {
      simulate(*root_);
    }

    return collect_root_result(*root_);
  }

  void advance_root(const Move& chosen_move, const State& new_state) {
    if (!root_) {
      root_ = std::make_unique<NodeT>(new_state);
      return;
    }
    for (auto& e : root_->edges) {
      if (e.move == chosen_move && e.child) {
        root_ = std::move(e.child);
        return;
      }
    }
    root_ = std::make_unique<NodeT>(new_state);
  }

  const Node<State, BackupStrategy>* root() const { return root_.get(); }

private:
  using NodeT = Node<State, BackupStrategy>;
  using EdgeT = Edge<State, BackupStrategy>;

  struct PathStep {
    NodeT* node = nullptr;
    EdgeT* edge = nullptr;
  };

  MCTSConfig cfg_;
  Model* model_;
  std::mt19937_64 rng_;
  std::unique_ptr<NodeT> root_;
  MinMaxStats min_max_;

  void simulate(NodeT& root) {
    std::vector<PathStep> path;
    path.reserve(256);

    NodeT* node = &root;

    while (true) {
      path.push_back({node, nullptr});

      if (node->state.is_terminal()) {
        BackupValue bv = BackupStrategy::make_leaf_value(
            node->state.terminal_value(node->state.current_player()),
            nullptr,
            node->state);
        backup(path, bv);
        return;
      }

      if (!node->expanded) {
        EvalT eval = model_->evaluate(node->state);
        expand_with_eval(*node, eval);
        BackupValue bv = BackupStrategy::make_leaf_value(eval.value, &eval.payload, node->state);
        backup(path, bv);
        return;
      }

      EdgeT* best = select_puct(*node);
      assert(best && "Expanded node must have edges or be terminal");

      if (!best->child) {
        State child_state = node->state;
        child_state.apply_move(best->move);
        best->child = std::make_unique<NodeT>(child_state);
      }

      path.back().edge = best;
      node = best->child.get();
    }
  }

  EdgeT* select_puct(NodeT& node) {
    const double sqrtN = std::sqrt(std::max(1, node.N));
    EdgeT* best = nullptr;
    double best_score = -std::numeric_limits<double>::infinity();

    for (auto& e : node.edges) {
      const double q = (e.N > 0) ? min_max_.normalize(e.Q) : 0.5;
      const double u = cfg_.cpuct * e.prior * (sqrtN / (1.0 + e.N));
      const double score = q + u;
      if (score > best_score) {
        best_score = score;
        best = &e;
      }
    }
    return best;
  }

  void expand_if_needed(NodeT& node) {
    if (node.state.is_terminal() || node.expanded) return;
    EvalT eval = model_->evaluate(node.state);
    expand_with_eval(node, eval);
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
};

// ---------------------------
// NEW: Async/pumpable MCTS task
// ---------------------------
//
// Async Model concept required by mcts_task:
//
// Model must provide:
//   using Ticket = ...; // cheap movable handle
//   Ticket submit(const State& s);
//   bool   poll(const Ticket& t, Evaluation<Move,Payload>* out);
//
// poll must be safe to call from multiple CPU threads (read-only).
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
      // Terminal: nothing to expand; treat as done.
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

    // 0) Ensure root expansion is submitted (like expand_if_needed in sync search)
    did_any |= ensure_root_submitted();

    // 1) Reap any completed evals (including root expansion)
    did_any |= reap_ready();

    // If root not expanded yet, we cannot run simulations.
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
      // Waiting: likely blocked by inflight leaves
      break;
    }

    // 3) Reap again
    did_any |= reap_ready();

    if (done()) return PumpStatus::Done;
    return did_any ? PumpStatus::Progress : PumpStatus::Waiting;
  }

private:
  struct PathStep {
    NodeT* node = nullptr;
    EdgeT* edge = nullptr;
  };

  struct Pending {
    NodeT* leaf = nullptr;
    std::vector<PathStep> path;
    Ticket ticket;
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

  bool ensure_root_submitted() {
    if (root_->state.is_terminal()) return false;
    if (root_->expanded) return false;
    if (root_eval_submitted_) return false;

    // Submit eval for root expansion (not counted as a simulation)
    root_ticket_ = model_->submit(root_->state);
    root_->inflight = true;
    root_eval_submitted_ = true;
    return true;
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

  SimStepStatus simulate_once() {
    // Optional throttle: avoid unbounded pending queue
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
        if (node->inflight) return SimStepStatus::Waiting;

        // Submit leaf eval and return immediately
        Ticket t = model_->submit(node->state);
        node->inflight = true;

        pending_.push_back(Pending{node, std::move(path), std::move(t)});
        return SimStepStatus::Submitted;
      }

      EdgeT* best = select_puct_skip_inflight(*node);
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

    // First: check root expansion ticket
    if (root_eval_submitted_ && !root_->expanded) {
      EvalT eval;
      if (model_->poll(root_ticket_, &eval)) {
        root_->inflight = false;
        expand_with_eval(*root_, eval);

        // Apply Dirichlet noise at root (like sync search)
        if (!root_noise_applied_ && cfg_.dirichlet_epsilon > 0.0 && !root_->edges.empty()) {
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
      expand_with_eval(*p.leaf, eval);

      BackupValue bv = BackupStrategy::make_leaf_value(eval.value, &eval.payload, p.leaf->state);
      backup(p.path, bv);
      completed_sims_ += 1;
      did = true;
    }

    return did;
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
};

} // namespace agmcts
