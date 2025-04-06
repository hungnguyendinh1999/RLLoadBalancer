#ifndef LINEARDECAYSCHEDULER_H
#define LINEARDECAYSCHEDULER_H
#include "DecayScheduler.h"

class LinearDecayScheduler : public DecayScheduler {
public:
    LinearDecayScheduler(float min, float max, float rate);
    float getValue(unsigned input) override;
};

#endif //LINEARDECAYSCHEDULER_H
