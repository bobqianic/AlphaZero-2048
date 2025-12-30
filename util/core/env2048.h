//
// Created by qianp on 27/12/2025.
//

#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace env2048 {

// 2048 tile encoding: each cell is a 4-bit exponent.
// 0 = empty, 1 = 2, 2 = 4, ... value = 1 << exp.
// Board is 4x4 packed into 64 bits in row-major order:
// cell (r,c) is at nibble index (r*4+c), nibble shift = (r*4+c)*4.

enum class Action : std::uint8_t { Up = 0, Right = 1, Down = 2, Left = 3 };
static constexpr int kNumActions = 4;

// Tiny fast RNG (SplitMix64). Deterministic, header-only, no <random>.
struct RNG {
  std::uint64_t state = 0x9E3779B97F4A7C15ull;

  RNG() = default;
  explicit RNG(std::uint64_t seed) { reseed(seed); }

  void reseed(std::uint64_t seed) {
    // Avoid the all-zero state.
    state = seed ? seed : 0x9E3779B97F4A7C15ull;
  }

  std::uint64_t next_u64() {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }

  // 32 random bits.
  std::uint32_t next_u32_raw() {
    return static_cast<std::uint32_t>(next_u64() >> 32);
  }

  // Uniform integer in [0, bound). bound must be > 0.
  std::uint32_t next_u32(std::uint32_t bound) {
    // Lemire's fast reduction using 32-bit source randomness.
    const std::uint64_t x = static_cast<std::uint64_t>(next_u32_raw());
    return static_cast<std::uint32_t>((x * bound) >> 32);
  }

  // Returns true with probability p = num/den (den>0, num<=den).
  bool bernoulli(std::uint32_t num, std::uint32_t den) {
    return next_u32(den) < num;
  }
};

struct StepResult {
  std::uint32_t reward = 0;  // merge reward from the move (no spawn reward)
  bool moved = false;        // whether the board changed
  bool terminal = false;     // no legal moves after the transition
};

// For chance-node expansion: applying a random spawn on a given board.
// Max outcomes is 2 * empty_cells <= 32.
struct ChanceOutcome {
  std::uint64_t board = 0;
  float prob = 0.0f;
};

namespace detail {

struct RowMove {
  std::uint16_t out = 0;        // new row (packed 4 nibbles)
  std::uint32_t reward = 0;     // reward for merges in that row
};

inline std::uint8_t get_nibble(std::uint64_t board, int idx) {
  return static_cast<std::uint8_t>((board >> (idx * 4)) & 0xFULL);
}

inline std::uint64_t set_nibble(std::uint64_t board, int idx, std::uint8_t v) {
  const std::uint64_t shift = static_cast<std::uint64_t>(idx * 4);
  const std::uint64_t mask = 0xFULL << shift;
  return (board & ~mask) | ((static_cast<std::uint64_t>(v) & 0xFULL) << shift);
}

// Reverse a packed row of 4 nibbles: [a b c d] -> [d c b a]
inline std::uint16_t reverse_row(std::uint16_t row) {
  return static_cast<std::uint16_t>(((row & 0x000FULL) << 12) |
                                    ((row & 0x00F0ULL) << 4)  |
                                    ((row & 0x0F00ULL) >> 4)  |
                                    ((row & 0xF000ULL) >> 12));
}

inline RowMove compute_left_row(std::uint16_t row) {
  std::uint8_t t[4] = {
    static_cast<std::uint8_t>( row        & 0xF),
    static_cast<std::uint8_t>((row >> 4)  & 0xF),
    static_cast<std::uint8_t>((row >> 8)  & 0xF),
    static_cast<std::uint8_t>((row >> 12) & 0xF),
  };

  // Compress (remove zeros)
  std::uint8_t comp[4] = {0,0,0,0};
  int n = 0;
  for (int i = 0; i < 4; ++i) if (t[i]) comp[n++] = t[i];

  // Merge adjacent equals
  std::uint8_t merged[4] = {0,0,0,0};
  int m = 0;
  std::uint32_t reward = 0;
  for (int i = 0; i < n; ++i) {
    if (i + 1 < n && comp[i] == comp[i + 1]) {
      const std::uint8_t v = static_cast<std::uint8_t>(comp[i] + 1);
      merged[m++] = v;
      // merged tile value is 2^v -> 1<<v in this exponent encoding
      reward += (v ? (1u << v) : 0u);
      ++i;
    } else {
      merged[m++] = comp[i];
    }
  }

  // Pack back into 16-bit row
  std::uint16_t out = 0;
  out |= static_cast<std::uint16_t>(merged[0] & 0xF) << 0;
  out |= static_cast<std::uint16_t>(merged[1] & 0xF) << 4;
  out |= static_cast<std::uint16_t>(merged[2] & 0xF) << 8;
  out |= static_cast<std::uint16_t>(merged[3] & 0xF) << 12;

  return RowMove{out, reward};
}

struct LUT {
  std::array<RowMove, 65536> left{};
};

// Built once per process (thread-safe since C++11).
inline const LUT& lut() {
  static const LUT table = []() {
    LUT t{};
    for (std::uint32_t r = 0; r <= 0xFFFFu; ++r) {
      t.left[r] = compute_left_row(static_cast<std::uint16_t>(r));
    }
    return t;
  }();
  return table;
}

inline RowMove apply_left(std::uint16_t row) {
  return lut().left[row];
}

inline RowMove apply_right(std::uint16_t row) {
  const std::uint16_t rev = reverse_row(row);
  RowMove mv = apply_left(rev);
  mv.out = reverse_row(mv.out);
  return mv;
}

// Extract a row (r in [0,3]) as 16-bit packed nibbles.
inline std::uint16_t get_row(std::uint64_t board, int r) {
  return static_cast<std::uint16_t>((board >> (r * 16)) & 0xFFFFULL);
}

// Set a row (r in [0,3]) from 16-bit packed nibbles.
inline std::uint64_t set_row(std::uint64_t board, int r, std::uint16_t row) {
  const std::uint64_t shift = static_cast<std::uint64_t>(r * 16);
  const std::uint64_t mask  = 0xFFFFULL << shift;
  return (board & ~mask) | (static_cast<std::uint64_t>(row) << shift);
}

// Extract a column (c in [0,3]) as 16-bit packed nibbles top->bottom.
inline std::uint16_t get_col(std::uint64_t board, int c) {
  std::uint16_t col = 0;
  for (int r = 0; r < 4; ++r) {
    const int idx = r * 4 + c;
    col |= static_cast<std::uint16_t>(get_nibble(board, idx) & 0xF) << (r * 4);
  }
  return col;
}

// Set a column (c in [0,3]) from 16-bit packed nibbles top->bottom.
inline std::uint64_t set_col(std::uint64_t board, int c, std::uint16_t col) {
  for (int r = 0; r < 4; ++r) {
    const int idx = r * 4 + c;
    const std::uint8_t v = static_cast<std::uint8_t>((col >> (r * 4)) & 0xF);
    board = set_nibble(board, idx, v);
  }
  return board;
}

} // namespace detail

class Env {
public:
  using Board = std::uint64_t;

  Env() = default;
  explicit Env(Board b) : board_(b) {}

  // --- Core accessors ---
  Board board() const { return board_; }
  void set_board(Board b) { board_ = b; }
  std::uint32_t score() const { return score_; } // optional tracked total reward
  void set_score(std::uint32_t s) { score_ = s; }

  // --- Reset / initialization ---
  // Typical 2048 starts with 2 tiles.
  void reset(RNG& rng, int initial_tiles = 2) {
    board_ = 0;
    score_ = 0;
    for (int i = 0; i < initial_tiles; ++i) {
      spawn_random(rng);
    }
  }

  // --- Deterministic move (no spawn) ---
  // Returns reward and whether board changed. Does NOT update terminal flag.
  StepResult move(Action a) {
    StepResult r{};
    const Board before = board_;
    Board after = before;

    switch (a) {
      case Action::Left:  after = move_left_(before, r.reward, r.moved); break;
      case Action::Right: after = move_right_(before, r.reward, r.moved); break;
      case Action::Up:    after = move_up_(before, r.reward, r.moved); break;
      case Action::Down:  after = move_down_(before, r.reward, r.moved); break;
      default: break;
    }

    if (r.moved) {
      board_ = after;
      score_ += r.reward;
    }
    // terminal computed by step() or caller via is_terminal().
    return r;
  }

  // --- Stochastic transition (move + random spawn if moved) ---
  StepResult step(Action a, RNG& rng) {
    StepResult r = move(a);
    if (r.moved) {
      spawn_random(rng);
    }
    r.terminal = is_terminal();
    return r;
  }

  // --- Spawning ---
  // Spawns a random tile (2 with p=0.9, 4 with p=0.1) into a random empty cell.
  // Returns false if no empty cells.
  bool spawn_random(RNG& rng) {
    std::array<int, 16> empties{};
    int n = 0;
    for (int i = 0; i < 16; ++i) {
      if (detail::get_nibble(board_, i) == 0) empties[n++] = i;
    }
    if (n == 0) return false;

    const int idx = empties[rng.next_u32(static_cast<std::uint32_t>(n))];
    const std::uint8_t exp = rng.bernoulli(1, 10) ? 2 : 1; // 10% -> 4, else 2
    board_ = detail::set_nibble(board_, idx, exp);
    return true;
  }

  // Enumerate all possible spawn outcomes on the CURRENT board, with probabilities.
  // Returns number of filled outcomes (<=32).
  std::size_t enumerate_spawns(std::array<ChanceOutcome, 32>& out) const {
    std::array<int, 16> empties{};
    int n = 0;
    for (int i = 0; i < 16; ++i) {
      if (detail::get_nibble(board_, i) == 0) empties[n++] = i;
    }
    if (n == 0) return 0;

    const float p_each = 1.0f / static_cast<float>(n);
    std::size_t k = 0;
    for (int ei = 0; ei < n; ++ei) {
      const int cell = empties[ei];

      // spawn 2 (exp=1), prob 0.9
      out[k++] = ChanceOutcome{detail::set_nibble(board_, cell, 1), 0.9f * p_each};
      // spawn 4 (exp=2), prob 0.1
      out[k++] = ChanceOutcome{detail::set_nibble(board_, cell, 2), 0.1f * p_each};
    }
    return k;
  }

  // --- Termination / legal actions ---
  bool is_terminal() const { return legal_actions_mask() == 0; }

  // Bitmask with bit(Action) set if that action would change the board.
  std::uint8_t legal_actions_mask() const {
    std::uint8_t mask = 0;
    if (can_move_(Action::Up))    mask |= (1u << static_cast<std::uint8_t>(Action::Up));
    if (can_move_(Action::Right)) mask |= (1u << static_cast<std::uint8_t>(Action::Right));
    if (can_move_(Action::Down))  mask |= (1u << static_cast<std::uint8_t>(Action::Down));
    if (can_move_(Action::Left))  mask |= (1u << static_cast<std::uint8_t>(Action::Left));
    return mask;
  }

  bool is_legal(Action a) const {
    return (legal_actions_mask() & (1u << static_cast<std::uint8_t>(a))) != 0;
  }

  // --- Feature helpers (useful for policies / eval) ---
  std::array<std::uint8_t, 16> to_exponents() const {
    std::array<std::uint8_t, 16> out{};
    for (int i = 0; i < 16; ++i) out[i] = detail::get_nibble(board_, i);
    return out;
  }

  // Max tile value on board (0 if empty).
  std::uint32_t max_tile() const {
    std::uint8_t mx = 0;
    for (int i = 0; i < 16; ++i) mx = std::max(mx, detail::get_nibble(board_, i));
    return mx ? (1u << mx) : 0u;
  }

  int empty_cells() const {
    int n = 0;
    for (int i = 0; i < 16; ++i) n += (detail::get_nibble(board_, i) == 0);
    return n;
  }

private:
  Board board_ = 0;
  std::uint32_t score_ = 0;

  bool can_move_(Action a) const {
    using namespace detail;
    const Board b = board_;

    if (a == Action::Left) {
      for (int r = 0; r < 4; ++r) {
        const std::uint16_t row = get_row(b, r);
        if (apply_left(row).out != row) return true;
      }
      return false;
    }

    if (a == Action::Right) {
      for (int r = 0; r < 4; ++r) {
        const std::uint16_t row = get_row(b, r);
        if (apply_right(row).out != row) return true;
      }
      return false;
    }

    if (a == Action::Up) {
      for (int c = 0; c < 4; ++c) {
        const std::uint16_t col = get_col(b, c);
        if (apply_left(col).out != col) return true;
      }
      return false;
    }

    // Down
    for (int c = 0; c < 4; ++c) {
      const std::uint16_t col = get_col(b, c);
      const std::uint16_t rev = reverse_row(col);
      RowMove mv = apply_left(rev);
      const std::uint16_t out = reverse_row(mv.out);
      if (out != col) return true;
    }
    return false;
  }

  static Board move_left_(Board b, std::uint32_t& reward, bool& moved) {
    using namespace detail;
    reward = 0;
    moved = false;
    Board out = 0;

    for (int r = 0; r < 4; ++r) {
      const std::uint16_t row = get_row(b, r);
      const RowMove mv = apply_left(row);
      reward += mv.reward;
      moved |= (mv.out != row);
      out |= static_cast<Board>(mv.out) << (r * 16);
    }
    return out;
  }

  static Board move_right_(Board b, std::uint32_t& reward, bool& moved) {
    using namespace detail;
    reward = 0;
    moved = false;
    Board out = 0;

    for (int r = 0; r < 4; ++r) {
      const std::uint16_t row = get_row(b, r);
      const RowMove mv = apply_right(row);
      reward += mv.reward;
      moved |= (mv.out != row);
      out |= static_cast<Board>(mv.out) << (r * 16);
    }
    return out;
  }

  static Board move_up_(Board b, std::uint32_t& reward, bool& moved) {
    using namespace detail;
    reward = 0;
    moved = false;
    Board out = b; // we'll overwrite columns

    for (int c = 0; c < 4; ++c) {
      const std::uint16_t col = get_col(b, c);
      const RowMove mv = apply_left(col);
      reward += mv.reward;
      moved |= (mv.out != col);
      out = set_col(out, c, mv.out);
    }
    return out;
  }

  static Board move_down_(Board b, std::uint32_t& reward, bool& moved) {
    using namespace detail;
    reward = 0;
    moved = false;
    Board out = b; // we'll overwrite columns

    for (int c = 0; c < 4; ++c) {
      const std::uint16_t col = get_col(b, c);
      const std::uint16_t rev = reverse_row(col);
      RowMove mv = apply_left(rev);
      const std::uint16_t new_col = reverse_row(mv.out);
      reward += mv.reward;
      moved |= (new_col != col);
      out = set_col(out, c, new_col);
    }
    return out;
  }
};

} // namespace env2048
