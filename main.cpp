// #include "Agent/QLAgent.h"
// #include "Agent/DPAgent.h"
#include "Agent/DQNAgent.h"
#include "DecayScheduler/LinearDecayScheduler.h"
#include "DecayScheduler/ExponentialDecayScheduler.h"
#include "Environment/Environment.h"
#include "Utils/Plot.h"

int main() {
    const unsigned seed = 123;
    const unsigned numProc = 4;
    const unsigned numThread = 4;
    const unsigned numTask = 40;
    const unsigned maxTaskDuration = 10;

    const float epsilonMin = 0.1;
    const float epsilonMax = 1.0;
    const float epsilonDecayRate = 0.001;

    const float alpha = 0.5;
    const float gamma = 0.99f;

    const float theta = 0.01;

    const unsigned numRun = 1;
    const unsigned numEpisode = 1000;

    // Data for plotting
    std::vector<std::vector<int>> rewards;

    // Running the train and rollout
    const auto env = std::make_shared<Environment>(numProc, numTask, numThread, maxTaskDuration, seed);
    const auto ds = std::make_shared<LinearDecayScheduler>(epsilonMin, epsilonMax, epsilonDecayRate);
    /**
    // Run DP
    auto dp = DPAgent(env, gamma, theta);
    dp.run_value_iteration();

    // Run Q Learning
    std::unique_ptr<QLAgent> agent;
    for (unsigned i = 0; i < numRun; i++) {
        agent = std::make_unique<QLAgent>(env, alpha, gamma, ds, seed);
        rewards.push_back(agent->train(numEpisode));
    }
    */
    // Run Deep Q-net
    /**
     * Variables for DQN agent
     */
    const auto ds_expo = std::make_shared<ExponentialDecayScheduler>(epsilonMin, epsilonMax, epsilonDecayRate);
    const float learning_rate = 1e-3f;
    int target_update_freq = 1000;
    size_t replay_capacity = 10000;
    float prepopulate_steps = 2500;
    size_t batch_size = 64;

    int state_size = env->reset().size();
    int action_size = env->getNumAction();
    std::vector<int> hidden_layers = {128, 128, 64}; // You can change this

    
    // // Create and train DQN agent
    std::unique_ptr<DQNAgent> agent;
    for (unsigned i = 0; i < numRun; i++) {
        agent = std::make_unique<DQNAgent>(env, state_size, action_size, hidden_layers, gamma, learning_rate,
            ds_expo, target_update_freq, replay_capacity,
            prepopulate_steps, batch_size);
        rewards.push_back(agent->train(numEpisode));
    }

    agent->rollout();

    Plot::AverageRewardsOverEpisodes(rewards);
    return 0;
}
