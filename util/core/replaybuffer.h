// replay.h
#pragma once
#include "step.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>
#include <vector>

// Fenwick from your trainer.cpp (unchanged)
struct Fenwick {
  int n = 0;
  std::vector<double> bit;
  explicit Fenwick(int n_) : n(n_), bit(n_ + 1, 0.0) {}

  void add(int idx, double delta) { for (int i = idx + 1; i <= n; i += i & -i) bit[i] += delta; }
  double sum_prefix(int idx) const { double s=0.0; for (int i=idx+1;i>0;i-=i&-i) s += bit[i]; return s; }
  double total() const { return (n>0) ? sum_prefix(n-1) : 0.0; }

  int lower_bound(double x) const {
    int idx=0; double cur=0.0;
    int step=1; while (step < n) step <<= 1;
    for (int k=step; k!=0; k>>=1) {
      int next = idx + k;
      if (next <= n && cur + bit[next] < x) { idx = next; cur += bit[next]; }
    }
    return std::min(idx, n-1);
  }
};

class ReplayBuffer {
public:
  ReplayBuffer(size_t capacity,
               double alpha,
               double beta,
               int nstep,
               double lambda,
               double discount)
    : cap_(capacity),
      alpha_(alpha),
      beta_(beta),
      nstep_(nstep),
      lambda_(lambda),
      discount_(discount),
      data_(capacity),
      fw_((int)capacity),
      weights_(capacity, 0.0) {}

  size_t size() const { return size_.load(std::memory_order_relaxed); }

  // Ingest an episode: compute TD(lambda) and initial priority for each step, then insert into ring.
  void push_episode(std::vector<Step>& ep) {
    if (ep.empty()) return;

    const int T = (int)ep.size();
    std::vector<std::uint32_t> rewards(T);
    std::vector<double> bootstrap_v(T + 1, 0.0);

    for (int t = 0; t < T; ++t) {
      rewards[t] = ep[t].reward;
      bootstrap_v[t] = (double)ep[t].root_value_raw;
    }
    bootstrap_v[T] = 0.0;

    // fill targets + priority in the episode vector itself
    for (int t = 0; t < T; ++t) {
      const double z = td_lambda_target_(rewards, bootstrap_v, t);
      const double v = bootstrap_v[t];
      ep[t].value_target = (float)z;
      ep[t].priority = (float)(std::abs(v - z) + 1e-6);
    }

    std::unique_lock<std::mutex> lk(mu_);
    for (int t = 0; t < T; ++t) {
      push_step_locked_(ep[t]);
    }
    lk.unlock();
    cv_.notify_all();
  }

  // Sample a batch: returns copied steps + indices + normalized importance weights
  void sample_batch(int batch,
                    std::mt19937_64& rng,
                    std::vector<Step>& out_steps,
                    std::vector<int>& out_indices,
                    std::vector<float>& out_isw) {
    out_steps.clear();
    out_indices.clear();
    out_isw.clear();
    out_steps.reserve((size_t)batch);
    out_indices.reserve((size_t)batch);
    out_isw.reserve((size_t)batch);

    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&]{
      return stop_ || size_.load(std::memory_order_relaxed) >= (size_t)std::max(1, batch);
    });
    if (stop_) return;

    const size_t N = size_.load(std::memory_order_relaxed);
    const double total_w = fw_.total();
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    float max_isw = 1e-12f;

    for (int b = 0; b < batch; ++b) {
      int idx = 0;
      if (total_w > 1e-12) {
        const double u = uni(rng) * total_w;
        idx = fw_.lower_bound(std::max(1e-12, u));
      } else {
        idx = (int)(uni(rng) * (double)N);
        idx = std::min(idx, (int)N - 1);
      }

      out_indices.push_back(idx);
      out_steps.push_back(data_[(size_t)idx]);

      const double wi = weights_[(size_t)idx];
      const double Pi = (total_w > 1e-12) ? (wi / total_w) : (1.0 / std::max<size_t>(1, N));
      const double isw = std::pow((1.0 / double(N)) * (1.0 / std::max(1e-12, Pi)), beta_);

      out_isw.push_back((float)isw);
      max_isw = std::max(max_isw, (float)isw);
    }

    for (auto& w : out_isw) w /= max_isw;
  }

  // Update PER priorities in-place
  void update_priorities(const std::vector<int>& indices,
                         const std::vector<float>& new_p) {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < indices.size(); ++i) {
      const int idx = indices[i];
      const double p = std::max(1e-6, (double)new_p[i]);

      const double oldp = (double)data_[(size_t)idx].priority;
      data_[(size_t)idx].priority = (float)p;

      const double oldw = weights_[(size_t)idx];
      const double neww = std::pow(p, alpha_);
      weights_[(size_t)idx] = neww;
      fw_.add(idx, neww - oldw);
    }
  }

  void request_stop() {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
    cv_.notify_all();
  }

private:
  size_t cap_;
  double alpha_;
  double beta_;
  int nstep_;
  double lambda_;
  double discount_;

  std::vector<Step> data_;
  Fenwick fw_;
  std::vector<double> weights_;

  size_t head_ = 0;
  std::atomic<size_t> size_{0};

  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;

  void push_step_locked_(const Step& s) {
    const int idx = (int)head_;

    // If overwriting, subtract old weight from Fenwick
    if (size_.load(std::memory_order_relaxed) == cap_) {
      const double oldw = weights_[head_];
      if (oldw != 0.0) fw_.add(idx, -oldw);
    } else {
      size_.store(size_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    }

    data_[head_] = s;

    const double p = std::max(1e-6, (double)s.priority);
    const double w = std::pow(p, alpha_);
    weights_[head_] = w;
    fw_.add(idx, w);

    head_ = (head_ + 1) % cap_;
  }

  double td_lambda_target_(const std::vector<std::uint32_t>& rewards,
                           const std::vector<double>& bootstrap_v,
                           int t) const {
    const int T = (int)rewards.size();

    auto nstep_return = [&](int n) -> double {
      double g = 0.0;
      double powd = 1.0;
      for (int k = 0; k < n; ++k) {
        const int idx = t + k;
        if (idx >= T) break;
        g += powd * double(rewards[idx]);
        powd *= discount_;
      }
      const int bidx = t + n;
      if (bidx < (int)bootstrap_v.size()) g += powd * bootstrap_v[bidx];
      return g;
    };

    double out = 0.0;
    double w = 1.0;
    for (int n = 1; n < nstep_; ++n) {
      out += (1.0 - lambda_) * w * nstep_return(n);
      w *= lambda_;
    }
    out += w * nstep_return(nstep_);
    return out;
  }
};
