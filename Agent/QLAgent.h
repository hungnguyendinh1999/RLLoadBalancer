#ifndef QLAGENT_H
#define QLAGENT_H
#include <random>
#include <torch/torch.h>
#include "BaseAgent.h"
#include "../DecayScheduler/DecayScheduler.h"
#include "../Environment/Environment.h"

class QLAgent : public BaseAgent {
public:
    QLAgent(const std::shared_ptr<Environment> &env, float alpha, float gamma, const std::shared_ptr<DecayScheduler> &decayScheduler);
    QLAgent(const std::shared_ptr<Environment> &env, float alpha, float gamma, const std::shared_ptr<DecayScheduler> &decayScheduler, unsigned seed);
    unsigned getBehaviorPolicy(std::vector<unsigned> s, unsigned t) override;
    unsigned getTargetPolicy(std::vector<unsigned> s) override;
    void update(std::vector<unsigned> s, unsigned a, int r, std::vector<unsigned> sPrime) override;
    std::vector<int> train(unsigned numEpisode);
    void rollout();
private:
    std::shared_ptr<Environment> _env;
    float _alpha;
    float _gamma;
    std::shared_ptr<DecayScheduler> _decayScheduler;
    std::mt19937 _randomizer;
    torch::Tensor _q;

    unsigned _argmax(const torch::Tensor& v);
    static std::vector<at::indexing::TensorIndex> _getIndicesTensor(const std::vector<unsigned> &s);
    static std::vector<at::indexing::TensorIndex> _getIndicesTensor(const std::vector<unsigned> &s, unsigned a);
};

#endif //QLAGENT_H
