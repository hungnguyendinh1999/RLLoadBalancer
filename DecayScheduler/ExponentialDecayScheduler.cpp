#include "ExponentialDecayScheduler.h"

// #include <ATen/ops/amax.h>
#include <algorithm>

ExponentialDecayScheduler::ExponentialDecayScheduler(const float min, const float max, const float rate)
    : DecayScheduler(min, max, rate) {}

float ExponentialDecayScheduler::getValue(const unsigned input) {
    return std::max(_min, _max * std::exp(-_rate * static_cast<float>(input)));
}