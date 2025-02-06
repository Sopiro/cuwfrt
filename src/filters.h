#pragma once

#include "api.cuh"
#include "common.h"

namespace cuwfrt
{

inline __cpu_gpu__ Point2 SampleBoxFilter(Float extent, const Point2& u)
{
    // Remap [0, 1]^2 to [-extent/2, extent/2]^2
    return (2 * u - 1) * (extent / 2);
}

inline __cpu_gpu__ Point2 SampleTentFilter(Float extent, const Point2& u)
{
    Float h = extent / 2;

    Float x = u[0] < Float(0.5) ? h * (sqrt(2 * u[0]) - 1) : h * (1 - sqrt(1 - 2 * (u[0] - Float(0.5))));
    Float y = u[1] < Float(0.5) ? h * (sqrt(2 * u[1]) - 1) : h * (1 - sqrt(1 - 2 * (u[1] - Float(0.5))));

    return Point2(x, y);
}

inline __cpu_gpu__ Point2 SampleGaussianFilter(Float sigma, const Point2& u)
{
    // Box Muller transform
    Float r = sigma * std::sqrt(-2 * std::log(fmax(u[0], Float(1e-8))));
    Float theta = two_pi * u[1];

    return Point2{ r * std::cos(theta), r * std::sin(theta) };
}

} // namespace cuwfrt
