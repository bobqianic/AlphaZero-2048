// main.cpp
#include "trainer.h"
#include "generator.h"
#include "util/core/replaybuffer.h"
#include "model.h"
#include "util/logger.h"
#include "play.h"

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <chrono>

namespace fs = std::filesystem;

// ----------------------- Path helpers -----------------------
static fs::path to_abs(fs::path p) {
  try {
    if (p.empty()) return p;
    if (p.is_relative()) p = fs::absolute(p);
    return p;
  } catch (...) {
    return p; // best-effort
  }
}

enum class RunMode { Train, Play };

static void usage() {
  std::cerr <<
    "main [--mode train|play] --ckpt <dir-or-file.pt> [--device cpu|cuda]\n"
    "     [--log <logfile>]\n"
    "     [--resume 0|1]\n"
    "\n"
    "PLAY MODE OPTIONS:\n"
    "     [--play_sims 800]\n"
    "     [--play_cpuct 1.5]\n"
    "     [--play_temp 0.0]\n"
    "     [--play_max_steps 1000000]\n"
    "     [--play_print 0|1]\n"
    "\n"
    "TRAIN MODE OPTIONS:\n"
    "     [--iters 100]\n"
    "     [--episodes_per_iter 50] [--sims 100] [--workers 2]\n"
    "     [--train_steps_per_iter 1000] [--batch 1024] [--lr 0.0003]\n"
    "     [--replay_capacity 125000]\n"
    "     [--alpha 1.0] [--beta 1.0] [--nstep 10] [--lambda 0.5] [--discount 0.999]\n"
    "     [--save_every_iter 1]\n"
    "     [--seed 1]\n";
}

// If ckpt_arg is a file => dir=parent, prefix=stem
// If ckpt_arg is a dir  => dir=ckpt_arg, prefix="ckpt"
static void resolve_ckpt_target(
  const fs::path& ckpt_arg,
  fs::path& out_dir,
  std::string& out_prefix
) {
  const fs::path p = ckpt_arg.empty() ? fs::path("checkpoints") : ckpt_arg;

  if (p.has_extension()) {
    out_dir = p.has_parent_path() ? p.parent_path() : fs::path(".");
    out_prefix = p.stem().string().empty() ? std::string("ckpt") : p.stem().string();
  } else {
    out_dir = p;
    out_prefix = "ckpt";
  }
}

static fs::path make_ckpt_path(
  const fs::path& dir,
  const std::string& prefix,
  std::uint64_t train_step,
  int iter
) {
  std::ostringstream name;
  name << prefix
       << "_step" << train_step
       << "_iter" << iter
       << ".pt";
  return dir / name.str();
}

static std::optional<fs::path> find_latest_checkpoint_in_dir(const fs::path& dir) {
  if (!fs::exists(dir) || !fs::is_directory(dir)) return std::nullopt;

  std::optional<fs::path> best;
  std::optional<fs::file_time_type> best_time;

  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    const fs::path p = entry.path();
    if (p.extension() != ".pt") continue;

    std::error_code ec;
    const auto t = fs::last_write_time(p, ec);
    if (ec) continue;

    if (!best || t > *best_time || (t == *best_time && p.string() > best->string())) {
      best = p;
      best_time = t;
    }
  }
  return best;
}

int main(int argc, char** argv) {
  RunMode mode = RunMode::Train;

  std::string ckpt_arg = "checkpoints";     // treated as dir-or-file.pt
  std::string log_arg  = "run.log";
  std::string device_str = "cuda";
  bool resume = true;

  // ---- Play mode options ----
  int play_sims = 800;
  double play_cpuct = 1.5;
  double play_temp = 0.0;
  int play_max_steps = 1000000;
  bool play_print = true;

  // ---- Train mode options ----
  int iters = 5000;

  int episodes_per_iter = 25000;
  int sims = 100;
  int workers = 12;

  int64_t train_steps_per_iter = 1000;
  int64_t batch = 1024;
  double lr = 3e-4;

  size_t replay_capacity = 125000 * 200;

  double alpha = 1.0;
  double beta  = 1.0;
  int nstep = 10;
  double lambda = 0.5;
  double discount = 0.999;

  int save_every_iter = 1;
  std::uint64_t seed = 1;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* name) {
      if (i + 1 >= argc) { std::cerr << "Missing value for " << name << "\n"; std::exit(2); }
      return std::string(argv[++i]);
    };

    if (a == "--mode") {
      std::string m = need("--mode");
      if (m == "train") mode = RunMode::Train;
      else if (m == "play") mode = RunMode::Play;
      else { std::cerr << "Unknown --mode " << m << "\n"; usage(); return 2; }
    } else if (a == "--ckpt") ckpt_arg = need("--ckpt");
    else if (a == "--log") log_arg = need("--log");
    else if (a == "--device") device_str = need("--device");
    else if (a == "--resume") resume = (std::stoi(need("--resume")) != 0);

    // play flags
    else if (a == "--play_sims") play_sims = std::max(1, std::stoi(need("--play_sims")));
    else if (a == "--play_cpuct") play_cpuct = std::stod(need("--play_cpuct"));
    else if (a == "--play_temp") play_temp = std::stod(need("--play_temp"));
    else if (a == "--play_max_steps") play_max_steps = std::max(1, std::stoi(need("--play_max_steps")));
    else if (a == "--play_print") play_print = (std::stoi(need("--play_print")) != 0);

    // train flags
    else if (a == "--iters") iters = std::stoi(need("--iters"));

    else if (a == "--episodes_per_iter") episodes_per_iter = std::stoi(need("--episodes_per_iter"));
    else if (a == "--sims") sims = std::stoi(need("--sims"));
    else if (a == "--workers") workers = std::max(1, std::stoi(need("--workers")));

    else if (a == "--train_steps_per_iter") train_steps_per_iter = std::stoll(need("--train_steps_per_iter"));
    else if (a == "--batch") batch = std::stoll(need("--batch"));
    else if (a == "--lr") lr = std::stod(need("--lr"));

    else if (a == "--replay_capacity") replay_capacity = (size_t)std::stoull(need("--replay_capacity"));

    else if (a == "--alpha") alpha = std::stod(need("--alpha"));
    else if (a == "--beta") beta = std::stod(need("--beta"));
    else if (a == "--nstep") nstep = std::stoi(need("--nstep"));
    else if (a == "--lambda") lambda = std::stod(need("--lambda"));
    else if (a == "--discount") discount = std::stod(need("--discount"));

    else if (a == "--save_every_iter") save_every_iter = std::max(0, std::stoi(need("--save_every_iter")));
    else if (a == "--seed") seed = std::stoull(need("--seed"));
    else { usage(); return 2; }
  }

  // Convert to absolute paths early
  fs::path log_path = to_abs(fs::path(log_arg));
  auto logger = std::make_shared<Logger>(log_path, /*max_queue=*/1u<<16);

  fs::path ckpt_dir;
  std::string ckpt_prefix;
  resolve_ckpt_target(fs::path(ckpt_arg), ckpt_dir, ckpt_prefix);
  ckpt_dir = to_abs(ckpt_dir);
  fs::create_directories(ckpt_dir);

  logf(*logger, "Log file: ", log_path.string());
  logf(*logger, "Checkpoint directory: ", ckpt_dir.string(), " (prefix=", ckpt_prefix, ")");

  torch::Device device = torch::kCPU;
  if (device_str == "cuda") {
    if (torch::cuda::is_available()) device = torch::kCUDA;
    else logf(*logger, "CUDA requested but not available; using CPU.");
  }
  logf(*logger, "Device: ", (device.is_cuda() ? "cuda" : "cpu"));

  // ---- Trainer owns model + optimizer state ----
  rl2048::Net net;
  TrainerConfig tcfg;
  tcfg.batch = batch;
  tcfg.lr = lr;
  tcfg.log_every = 100;

  Trainer trainer(net, device, tcfg, logger);

  std::uint64_t train_step = 0;

  // ---- Resume / Play ----
  if (resume || mode == RunMode::Play) {
    fs::path ckpt_input = to_abs(fs::path(ckpt_arg));
    bool loaded = false;

    if (ckpt_input.has_extension() && fs::exists(ckpt_input) && fs::is_regular_file(ckpt_input)) {
      loaded = trainer.load_checkpoint(ckpt_input.string(), train_step);
      logf(*logger, "Resume: tried file ", ckpt_input.string(), " -> ", (loaded ? "OK" : "FAILED"));
    } else {
      auto latest = find_latest_checkpoint_in_dir(ckpt_dir);
      if (latest) {
        loaded = trainer.load_checkpoint(latest->string(), train_step);
        logf(*logger, "Resume: latest in dir ", ckpt_dir.string(), " is ", latest->string(),
             " -> ", (loaded ? "OK" : "FAILED"));
      } else {
        logf(*logger, "Resume: no checkpoints found in ", ckpt_dir.string(), " (starting fresh)");
      }
    }
  }

  // =========================
  // PLAY MODE
  // =========================
  if (mode == RunMode::Play) {
    logf(*logger, "Mode: play");

    play2048::PlayConfig pcfg;
    pcfg.sims = play_sims;
    pcfg.cpuct = play_cpuct;
    pcfg.temperature = play_temp;
    pcfg.max_steps = play_max_steps;
    pcfg.print_each_step = play_print;
    pcfg.seed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()
                ).count();

    logf(*logger, "Play cfg: sims=", pcfg.sims,
         " cpuct=", pcfg.cpuct,
         " temp=", pcfg.temperature,
         " max_steps=", pcfg.max_steps,
         " print=", (pcfg.print_each_step ? 1 : 0),
         " seed=", pcfg.seed);

    trainer.net()->eval();
    play2048::play(trainer.net(), device, pcfg, std::cout);
    return 0;
  }

  // =========================
  // TRAIN MODE
  // =========================
  logf(*logger, "Mode: train");

  // ---- Replay buffer (in-memory) ----
  ReplayBuffer replay(
    replay_capacity,
    /*alpha=*/alpha,
    /*beta=*/beta,
    /*nstep=*/nstep,
    /*lambda=*/lambda,
    /*discount=*/discount
  );

  // ---- Generator (uses SAME underlying net params via ModuleHolder sharing) ----
  GeneratorConfig gcfg;
  gcfg.episodes = episodes_per_iter;
  gcfg.sims = sims;
  gcfg.workers = workers;
  gcfg.seed = seed;

  Generator generator(trainer.net(), device, gcfg, logger);

  // ---- Loop: generate -> train -> generate ----
  for (int iter = 0; iter < iters; ++iter) {
    logf(*logger, "=== ITER ", iter, " (train_step=", train_step,
         ", replay_size=", replay.size(), ") ===");

    generator.generate_into(replay, train_step);
    while (replay.size() < (size_t)batch) {
      logf(*logger, "Replay has only ", replay.size(),
           " steps; generating more to reach batch=", batch);
      generator.generate_into(replay, train_step);
    }

    trainer.net()->train();
    trainer.train_steps(replay, train_steps_per_iter, train_step);

    if (save_every_iter > 0 && (iter % save_every_iter) == 0) {
      const fs::path out = make_ckpt_path(ckpt_dir, ckpt_prefix, train_step, iter);
      trainer.save_checkpoint(out.string(), train_step);
      logf(*logger, "Saved checkpoint: ", out.string());
    }
  }

  // Final checkpoint (also non-overwriting)
  {
    const fs::path out = make_ckpt_path(ckpt_dir, ckpt_prefix, train_step, iters);
    trainer.save_checkpoint(out.string(), train_step);
    logf(*logger, "Saved final checkpoint: ", out.string());
  }

  logf(*logger, "Done. Final train_step=", train_step, " replay_size=", replay.size());
  return 0;
}
