#ifndef REPLAYBUFFER_H
#define REPLAYBUFFER_H

#include <deque>
#include <random>
#include <tuple>
#include <vector>

#include "../Environment/Environment.h"

struct Batch {
    std::vector<unsigned> state;
    unsigned action;
    float reward;
    std::vector<unsigned> next_state;
    bool done;
};

class ReplayBuffer {
public:
    ReplayBuffer(size_t max_size);
    ReplayBuffer(size_t max_size, unsigned seed);
    ReplayBuffer() : ReplayBuffer(10000) {}  // default constructor
    void add(std::vector<unsigned> state, unsigned action, float reward, std::vector<unsigned> next_state, bool done);
    std::vector<Batch> sample(size_t batch_size);
    void populate(const std::shared_ptr<Environment>& env, size_t num_steps);

    size_t get_size() const;
private:
    std::deque<Batch> buffer_;
    size_t capacity_;
    std::mt19937 _randomizer;
};
#endif // REPLAYBUFFER_H