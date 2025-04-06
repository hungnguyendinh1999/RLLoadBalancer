#include "Plot.h"

#include <matplot/matplot.h>
#include <torch/torch.h>

void Plot::AverageRewardsOverEpisodes(const std::vector<std::vector<int>>& data) {
    unsigned num_runs = data.size();
    unsigned num_episodes = data[0].size();

    const torch::Tensor tensor_data = torch::zeros({num_runs, num_episodes});

    for (int i = 0; i < num_runs; ++i) {
        tensor_data[i] = torch::tensor(data[i], torch::kFloat32);
    }

    const torch::Tensor mean_rewards = torch::mean(tensor_data, 0);  // Mean along the first dimension

    const std::vector<float> avg_rewards(mean_rewards.data_ptr<float>(), mean_rewards.data_ptr<float>() + num_episodes);

    // Plot the average reward curve
    matplot::plot(avg_rewards);
    matplot::xlabel("Episodes");
    matplot::ylabel("Average Reward");
    matplot::title("Q-Learning Average Reward Over Time");
    matplot::show();
}
