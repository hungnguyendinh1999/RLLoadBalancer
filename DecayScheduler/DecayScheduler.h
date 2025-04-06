#ifndef DECAYSCHEDULER_H
#define DECAYSCHEDULER_H

class DecayScheduler {
public:
    DecayScheduler(float min, float max, float rate);
    virtual ~DecayScheduler() = default;
    virtual float getValue(unsigned input) = 0;
protected:
    float _min;
    float _max;
    float _rate;
};

#endif //DECAYSCHEDULER_H
