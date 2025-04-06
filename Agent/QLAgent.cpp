#include "QLAgent.h"

#include <iostream>
#include "../Utils/ProgressBar.h"

QLAgent::QLAgent(const std::shared_ptr<Environment> &env, const float alpha, const float gamma,
                    const std::shared_ptr<DecayScheduler> &decayScheduler) {
    this->_env = env;
    this->_alpha = alpha;
    this->_gamma = gamma;
    this->_decayScheduler = decayScheduler;
    this->_randomizer = std::mt19937(std::random_device()());

    std::vector<int64_t> qShape = {env->getNumTask() + 1};

    for (int proc = 0; proc < env->getNumProc(); ++proc) {
        qShape.push_back(env->getMaxThread() + 1);
    }

    qShape.push_back(env->getNumAction());

    this->_q = torch::full(qShape, 10.0f, torch::TensorOptions().dtype(torch::kFloat));
}

QLAgent::QLAgent(const std::shared_ptr<Environment> &env, const float alpha, const float gamma,
                    const std::shared_ptr<DecayScheduler> &decayScheduler, const unsigned seed) {
    this->_env = env;
    this->_alpha = alpha;
    this->_gamma = gamma;
    this->_decayScheduler = decayScheduler;
    this->_randomizer = std::mt19937(seed);

    std::vector<int64_t> qShape = {env->getNumTask() + 1};

    for (int proc = 0; proc < env->getNumProc(); ++proc) {
        qShape.push_back(env->getMaxThread() + 1);
    }

    qShape.push_back(env->getNumAction());

    this->_q = torch::full(qShape, 10.0f, torch::TensorOptions().dtype(torch::kFloat));
}

unsigned QLAgent::getBehaviorPolicy(const std::vector<unsigned> s, const unsigned t) {
    auto randChance = std::uniform_real_distribution<float>(0, 1);

    float chance = randChance(this->_randomizer);
    if (chance < this->_decayScheduler->getValue(t)) {
        auto randAllAction = std::uniform_int_distribution<unsigned>(0, this->_env->getNumAction() - 1);
        return randAllAction(this->_randomizer);
    }

    return getTargetPolicy(s);
}

unsigned QLAgent::getTargetPolicy(const std::vector<unsigned> s) {
    const torch::Tensor qs = this->_q.index(this->_getIndicesTensor(s));
    return _argmax(qs);
}

void QLAgent::update(const std::vector<unsigned> s, const unsigned a, const int r, const std::vector<unsigned> sPrime) {
    const unsigned bestAPrime = getTargetPolicy(sPrime);
    const auto nextQ = this->_q.index(this->_getIndicesTensor(sPrime, bestAPrime)).item<float>();
    const auto currentQ = this->_q.index(this->_getIndicesTensor(s, a)).item<float>();

    this->_q.index(this->_getIndicesTensor(s, a)) += this->_alpha * (r + nextQ - currentQ);
}

std::vector<int> QLAgent::train(const unsigned numEpisode) {
    this->_env->setDebug(false);
    std::vector<int> rewards = {};
    auto pb = ProgressBar("Training", numEpisode, [this, &rewards](const unsigned it) {
        std::vector<unsigned> s = this->_env->reset();
        bool done = false;
        unsigned a = getBehaviorPolicy(s, it);
        int episodeRewards = 0;
        while (!done) {
            int r = 0;
            std::vector<unsigned> sPrime;
            std::tie(sPrime, r, done) = this->_env->step(a);
            const unsigned aPrime = getBehaviorPolicy(s, it);

            this->update(s, a, r, sPrime);

            episodeRewards += r;
            s = sPrime;
            a = aPrime;
        }

        rewards.push_back(episodeRewards);
    });

    return rewards;
}

void QLAgent::rollout() {
    std::vector<unsigned> s = this->_env->reset();
    this->_env->setDebug(true);
    bool done = false;
    unsigned a = getTargetPolicy(s);
    unsigned t = 0;
    while (!done) {
        int r = 0;
        std::vector<unsigned> sPrime;
        std::tie(sPrime, r, done) = this->_env->step(a);
        const unsigned aPrime = getTargetPolicy(s);

        this->update(s, a, r, sPrime);

        s = sPrime;
        a = aPrime;
        ++t;
    }
    std::cout << "Took " << t << " time steps to finish!" << std::endl;
}

unsigned QLAgent::_argmax(const torch::Tensor& v) {
    const auto maxVal = v.max().item<float>();
    std::vector<unsigned> maxIndices = {};

    for (int i = 0; i < v.sizes()[0]; i++) {
        if (v[i].item<float>() == maxVal) {
            maxIndices.push_back(i);
        }
    }

    // Randomly choose among the max indices
    std::uniform_int_distribution<unsigned> dist(0, maxIndices.size() - 1);

    return maxIndices[dist(this->_randomizer)];
}

std::vector<at::indexing::TensorIndex> QLAgent::_getIndicesTensor(const std::vector<unsigned> &s) {
    std::vector<at::indexing::TensorIndex> shape = {};

    for (int i = 0; i < s.size(); ++i) {
        shape.push_back(at::indexing::TensorIndex(static_cast<int64_t>(s[i])));
    }

    return shape;
}

std::vector<at::indexing::TensorIndex> QLAgent::_getIndicesTensor(const std::vector<unsigned> &s, const unsigned a) {
    std::vector<at::indexing::TensorIndex> shape = {};

    for (int i = 0; i < s.size(); ++i) {
        shape.push_back(at::indexing::TensorIndex(static_cast<int64_t>(s[i])));
    }

    shape.push_back(at::indexing::TensorIndex(static_cast<int64_t>(a)));

    return shape;
}


