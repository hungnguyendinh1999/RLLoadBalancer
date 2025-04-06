#include "LinearDecayScheduler.h"

#include <ATen/ops/amax.h>


LinearDecayScheduler::LinearDecayScheduler(const float min, const float max, const float rate): DecayScheduler(min, max, rate) {}

float LinearDecayScheduler::getValue(const unsigned input) {
    return std::ranges::max(this->_min, this->_max - this->_rate * static_cast<float>(input));
}

