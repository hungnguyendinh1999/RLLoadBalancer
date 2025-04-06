#include "DecayScheduler.h"

DecayScheduler::DecayScheduler(const float min, const float max, const float rate) {
    this->_min = min;
    this->_max = max;
    this->_rate = rate;
}
