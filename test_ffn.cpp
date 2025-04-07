#include <iostream>
#include <torch/torch.h>
#include "Agent/FFN.h"

int main() {
    int input_size = 4;
    int output_size = 3;
    std::vector<int> hidden = {16, 16};

    FFN model = nullptr;;
    model = FFN(input_size, output_size, hidden);
    model->eval();  // Evaluation mode (optional)

    // Dummy input: batch of 2 states
    torch::Tensor input = torch::tensor({{1.0, 2.0, 3.0, 4.0},
                                         {4.0, 3.0, 2.0, 1.0}}, torch::kFloat32);

    torch::Tensor output = model->forward(input);
    std::cout << "TOutput Q-values:\n" << output << "\n" << std::endl;
    return 0;
}
