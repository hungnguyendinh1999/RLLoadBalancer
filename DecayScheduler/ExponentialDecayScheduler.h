#ifndef EXPONENTIALDECAYSCHEDULER_H
#define EXPONENTIALDECAYSCHEDULER_H
#include "DecayScheduler.h"

/**
 * @brief Epsilon scheduler that applies exponential decay:
 *        epsilon(t) = max(min, max * exp(-rate * t))
 */
class ExponentialDecayScheduler : public DecayScheduler {
public:
    ExponentialDecayScheduler(float min, float max, float rate);
    float getValue(unsigned input) override;
};

#endif //EXPONENTIALDECAYSCHEDULER_H
