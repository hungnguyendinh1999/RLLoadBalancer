#include "Agent/QLAgent.h"
#include "Agent/DPAgent.h"
#include "DecayScheduler/LinearDecayScheduler.h"
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
    const float gamma = 1.0;

    const float theta = 0.01;

    const unsigned numRun = 10;
    const unsigned numEpisode = 100000;

    // Data for plotting
    std::vector<std::vector<int>> rewards;

    // Running the train and rollout
    const auto env = std::make_shared<Environment>(numProc, numTask, numThread, maxTaskDuration, seed);
    const auto ds = std::make_shared<LinearDecayScheduler>(epsilonMin, epsilonMax, epsilonDecayRate);

    // Run DP
    auto dp = DPAgent(env, gamma, theta);
    dp.run_value_iteration();

    // Run Q Learning
    std::unique_ptr<QLAgent> agent;
    for (unsigned i = 0; i < numRun; i++) {
        agent = std::make_unique<QLAgent>(env, alpha, gamma, ds, seed);
        rewards.push_back(agent->train(numEpisode));
    }

    agent->rollout();

    Plot::AverageRewardsOverEpisodes(rewards);
    return 0;
}
