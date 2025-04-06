#include "FFN.h"

FFNImpl::FFNImpl(int input_size, int output_size, const std::vector<int>& hidden_layers) {
    int in_features = input_size;

    for (size_t i = 0; i < hidden_layers.size(); ++i) {
        int out_features = hidden_layers[i];
        layers->push_back(register_module("fc" + std::to_string(i),
                          torch::nn::Linear(in_features, out_features)));
        in_features = out_features;
    }

    // Final output layer
    layers->push_back(register_module("fc_out", torch::nn::Linear(in_features, output_size)));
}

torch::Tensor FFNImpl::forward(torch::Tensor x) {
    for (size_t i = 0; i < layers->size() - 1; ++i) {
        x = torch::relu(layers[i]->as<torch::nn::Linear>()->forward(x));
    }
    // Final layer: output Q-values without activation
    x = layers[layers->size() - 1]->as<torch::nn::Linear>()->forward(x);
    return x;
}
