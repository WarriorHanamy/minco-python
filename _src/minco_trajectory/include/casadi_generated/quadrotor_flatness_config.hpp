#pragma once

#include "flatness.hpp"

namespace minco::flatness::casadi_generated
{

inline constexpr DefaultConfig kEmbeddedConfig{
    .mass            = 1.1000000000000001,
    .gravity         = 9.8100000000000005,
    .horizontal_drag = 0.050000000000000003,
    .vertical_drag   = 0.050000000000000003,
    .parasitic_drag  = 0.01,
    .speed_smooth    = 0.001,
};

}  // namespace minco::flatness::casadi_generated

