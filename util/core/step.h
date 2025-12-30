// step.h
#pragma once
#include <array>
#include <cstdint>

struct Step {
    std::uint64_t board = 0;
    std::uint8_t action = 0;

    // Targets / labels
    std::uint32_t reward = 0;              // raw immediate reward after action
    std::array<float,4> pi{};              // search policy target

    // Bootstrap coming from MCTS (generator writes this)
    float root_value_raw = 0.0f;           // V(s_t) estimate (raw)

    // Trainer-derived (filled when episode is ingested into replay)
    float value_target = 0.0f;             // TD(lambda) raw return target
    float priority = 1.0f;                 // PER priority (abs(td_error)+eps), float is enough
};
