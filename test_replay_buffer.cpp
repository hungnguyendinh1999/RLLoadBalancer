#include <iostream>
#include "Agent/ReplayBuffer.h"
#include "Environment/Environment.h"  // Required to compile populate()

int main() {
    size_t capacity = 10;
    ReplayBuffer buffer(capacity);

    // Manually add transitions
    for (int i = 0; i < 5; ++i) {
        std::vector<unsigned> state = {static_cast<unsigned>(i), static_cast<unsigned>(i+1)};
        unsigned action = i % 2;
        float reward = i * 0.5f;
        std::vector<unsigned> next_state = {static_cast<unsigned>(i+1), static_cast<unsigned>(i+2)};
        bool done = (i == 4);
        buffer.add(state, action, reward, next_state, done);
    }

    std::cout << "ReplayBuffer size: " << buffer.get_size() << std::endl;

    auto sample = buffer.sample(3);
    for (size_t i = 0; i < sample.size(); ++i) {
        const auto& b = sample[i];
        std::cout << "Sample " << i << ": s=[" << b.state[0] << "," << b.state[1]
                  << "] a=" << b.action << " r=" << b.reward
                  << " done=" << b.done << std::endl;
    }

    return 0;
}
