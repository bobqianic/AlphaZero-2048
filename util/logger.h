// logger.h
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

namespace fs = std::filesystem;

class Logger {
public:
  // max_queue: upper bound of pending messages. When full, we drop (no blocking).
  explicit Logger(const fs::path& file_path,
                  std::size_t max_queue = 1u << 16 /* 65536 */)
      : max_queue_(max_queue) {
    fs::create_directories(file_path.parent_path());
    file_.open(file_path, std::ios::out | std::ios::app);
    if (!file_) {
      std::cerr << "Failed to open log file: " << file_path << "\n";
      std::exit(2);
    }

    writer_ = std::thread([this] { writer_loop_(); });
  }

  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  ~Logger() {
    stop();
  }

  // Enqueue a line. Returns false if dropped due to full queue or stopping.
  bool write_line(std::string line) {
    if (!accepting_.load(std::memory_order_relaxed)) return false;

    Item it{timestamp_utc_(), std::move(line)};

    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!accepting_.load(std::memory_order_relaxed)) return false;

      if (queue_.size() >= max_queue_) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return false; // drop instead of blocking
      }
      queue_.push_back(std::move(it));
    }
    cv_.notify_one();
    return true;
  }

  // Optional: wait until everything currently queued is written (this DOES block).
  void flush() {
    std::unique_lock<std::mutex> lk(mu_);
    flushed_cv_.wait(lk, [&] { return queue_.empty() && !writing_.load(); });
    // writer thread flushes the file when idle
  }

  void stop() {
    bool expected = true;
    if (!accepting_.compare_exchange_strong(expected, false)) {
      // already stopping/stopped
      return;
    }

    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_one();
    if (writer_.joinable()) writer_.join();
  }

private:
  struct Item {
    std::string ts;
    std::string msg;
  };

  static std::string timestamp_utc_() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S UTC");
    return oss.str();
  }

  void writer_loop_() {
    std::deque<Item> local;

    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });

        if (queue_.empty() && stop_) break;

        local.swap(queue_);
        writing_.store(true);
      }

      // Emit a summary if we dropped messages due to overload
      const auto dropped = dropped_.exchange(0, std::memory_order_relaxed);
      if (dropped > 0) {
        const auto ts = timestamp_utc_();
        write_impl_(ts, "[logger] dropped " + std::to_string(dropped) +
                           " messages (queue full)");
      }

      for (auto& it : local) {
        write_impl_(it.ts, it.msg);
      }
      local.clear();

      // Flush when idle to reduce data loss without flushing on every call
      file_.flush();

      {
        std::lock_guard<std::mutex> lk(mu_);
        writing_.store(false);
        if (queue_.empty()) flushed_cv_.notify_all();
      }
    }

    // Final drain (best-effort)
    file_.flush();
  }

  void write_impl_(const std::string& ts, const std::string& msg) {
    // Single thread performs all I/O => no interleaving and no extra locks here
    std::cout << ts << " " << msg << "\n";
    file_ << ts << " " << msg << "\n";
  }

  const std::size_t max_queue_;

  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable flushed_cv_;
  std::deque<Item> queue_;

  std::ofstream file_;
  std::thread writer_;

  std::atomic<bool> accepting_{true};
  bool stop_ = false;

  std::atomic<std::uint64_t> dropped_{0};
  std::atomic<bool> writing_{false};
};

// helper formatter (same style you already use)
template <typename... Args>
inline void logf(Logger& logger, Args&&... args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  (void)logger.write_line(oss.str());
}
