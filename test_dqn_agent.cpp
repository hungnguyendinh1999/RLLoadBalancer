#include <iostream>
#include <memory>
#include <vector>

#include "Agent/DQNAgent.h"
#include "DecayScheduler/ExponentialDecayScheduler.h"
#include "Environment/Environment.h"

int main() {
    std::cout << "[Test] Initializing environment..." << std::endl;

    unsigned num_processors = 2;
    unsigned num_tasks = 4;
    unsigned max_threads = 2;
    unsigned max_duration = 5;
    auto env = std::make_shared<Environment>(num_processors, num_tasks, max_threads, max_duration);

    std::cout << "[Test] Creating decay scheduler..." << std::endl;

    auto decayScheduler = std::make_shared<ExponentialDecayScheduler>(0.1f, 1.0f, 0.01f);

    int state_size = env->reset().size();
    int action_size = env->getNumAction();
    std::vector<int> hidden_layers = {64, 64, 64};

    float gamma = 0.95f;
    float lr = 1e-3f;
    int target_update_freq = 100;
    size_t replay_capacity = 1000;
    float prepopulate_steps = 200;
    size_t batch_size = 32;

    std::cout << "[Test] Creating DQN agent..." << std::endl;

    DQNAgent agent(env, state_size, action_size, hidden_layers, gamma, lr,
                   decayScheduler, target_update_freq, replay_capacity,
                   prepopulate_steps, batch_size);

    std::cout << "[Test] Starting short training..." << std::endl;

    unsigned episodes = 5;
    auto rewards = agent.train(episodes);

    std::cout << "[Test] Training finished. Episode rewards:\n";
    for (size_t i = 0; i < rewards.size(); ++i) {
        std::cout << "Episode " << i + 1 << ": " << rewards[i] << std::endl;
    }

    std::cout << "[Test] Running greedy rollout...\n";
    agent.rollout();

    std::cout << "[Test] Test complete ✅" << std::endl;
    return 0;
}
