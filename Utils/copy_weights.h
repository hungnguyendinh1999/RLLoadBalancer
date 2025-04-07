#ifndef COPY_WEIGHTS_H
#define COPY_WEIGHTS_H

#include <torch/torch.h>
#include <sstream>

/**
 * @brief Copies weights from one module to another using in-memory stream serialization.
 * 
 * This is useful when load_state_dict() is unavailable or if you're working
 * with ModuleHolder wrappers like FFN, Sequential, etc.
 * 
 * @tparam ModuleType A torch::nn::ModuleHolder<T> (e.g., FFN, torch::nn::Sequential, etc.)
 * @param from The source module (e.g., _q_net)
 * @param to   The target module to copy into (e.g., _target_net)
 */
template <typename ModuleType>
void copy_weights(const ModuleType& from, ModuleType& to) {
    std::stringstream buffer;
    torch::save(from, buffer);
    torch::load(to, buffer);
}

#endif // COPY_WEIGHTS_H
