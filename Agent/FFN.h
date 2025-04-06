#ifndef FFN_H
#define FFN_H

#include <torch/torch.h>

/**
 * Usage example: 
 * ```
 * std::vector<int> hidden_layers = {128, 128};  // Two hidden layers
 * FFN q_net(num_state, num_actions, hidden_layers);
 * ```
 */
class FFNImpl : public torch::nn::Module {
public:
    FFNImpl(int input_size, int output_size, 
            const std::vector<int>& hidden_layers);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::ModuleList layers;
};

TORCH_MODULE(FFN);

#endif // FFN_H
