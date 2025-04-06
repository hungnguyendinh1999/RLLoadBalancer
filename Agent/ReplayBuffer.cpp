#include "ReplayBuffer.h"
#include <algorithm>

ReplayBuffer::ReplayBuffer(size_t capacity)
    : capacity_(capacity), 
    _randomizer(std::mt19937(std::random_device()())) {}

ReplayBuffer::ReplayBuffer(size_t capacity, const unsigned seed)
    : capacity_(capacity),
    _randomizer(std::mt19937(seed)) {}

void ReplayBuffer::add(std::vector<unsigned> state, unsigned action, float reward, std::vector<unsigned> next_state, bool done) {
    Batch batch{state, action, static_cast<float>(reward), next_state, done};
    // <deque> pop_front and push_back are O(1)
    if (buffer_.size() >= capacity_) {
        buffer_.pop_front();
    }
    buffer_.push_back(batch);
}

std::vector<Batch> ReplayBuffer::sample(size_t batch_size) {
    std::vector<Batch> minibatch;
    std::sample(buffer_.begin(), buffer_.end(), std::back_inserter(minibatch),
                batch_size, _randomizer);
    return minibatch;
}

size_t ReplayBuffer::get_size() const {
    return buffer_.size();
}

void ReplayBuffer::populate(const std::shared_ptr<Environment>& env, size_t num_steps) {
    std::uniform_int_distribution<unsigned> action_dist(0, env->getNumAction() - 1);

    std::vector<unsigned> state = env->reset();
    for (size_t i = 0; i < num_steps; ++i) {
        unsigned action = action_dist(_randomizer);
        auto [next_state, reward, done] = env->step(action);

        this->add(state, action, reward, next_state, done);

        state = done ? env->reset() : next_state;
    }
}
