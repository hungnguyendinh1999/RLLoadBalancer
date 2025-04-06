#include "DQNAgent.h"
#include <numeric> // 
#include <algorithm> // max_element

DQNAgent::DQNAgent(
    const std::shared_ptr<Environment> &env, 
    int state_size, 
    int action_size, 
    const std::vector<int>& hidden_layers,
    float gamma, float lr,
    const std::shared_ptr<DecayScheduler> &decayScheduler,
    int target_update_freq, // 10K
    size_t replay_size,
    float replay_prepopulate_steps,
    size_t batch_size
) 
    : _env(env), 
    _decay_scheduler(decayScheduler), 
    _state_size(state_size),
    _action_size(action_size),
    _gamma(gamma),
    _target_update_freq(target_update_freq),
    _batch_size(batch_size)
    {
        this->_randomizer = std::mt19937(std::random_device()());
        // All the more complex private properties
        // init 2 networks - Q and target net and make sure they have the same weights
        this->_q_net = FFN(state_size, action_size, hidden_layers);
        this->_target_net = FFN(state_size, action_size, hidden_layers);
        target_net_->load_state_dict(q_net_->state_dict());
        target_net_->eval(); // target not in training mode

        this->optimizer = torch::optim::Adam(q_net_->parameters(), lr);
        
        // Prepopulate buffer using a random policy
        this->_replay_buffer = ReplayBuffer(replay_size);
        replay_buffer_.populate(env_, static_cast<size_t>(replay_prepopulate_steps));
    }

unsigned DQNAgent::getBehaviorPolicy(const std::vector<unsigned> s, const unsigned t) {
    std::uniform_real_distribution<float> randChance(0.0f, 1.0f);
    float epsilon = this->_decay_scheduler->getValue(t);

    float chance = randChance(this->_randomizer);
    if (chance < this->_decayScheduler->getValue(t)) {
        std::uniform_int_distribution<unsigned> randAllAction(0, this->_env->getNumAction() - 1);
        return randAllAction(this->_randomizer);
    }

    return getTargetPolicy(s); 
}

unsigned DQNAgent::getTargetPolicy(std::vector<unsigned> s) {
    // Convert the state vector to a tensor and add batch dimension
    torch::Tensor state_tensor = torch::tensor(s, torch::kFloat32).unsqueeze(0); // [1, state_size]

    // Get Q-values from the network and remove batch dimension
    torch::Tensor qs = this->_q_net->forward(state_tensor).squeeze(0); // [action_size]

    return _argmax(qs);
}

void DQNAgent::update(std::vector<unsigned> s, unsigned a, int r, std::vector<unsigned> sPrime) {
    bool done = sPrime.empty();
    _replay_buffer.add(s, a, static_cast<float>(r), sPrime, done);

    // Perform batch sampling and update Q-net
    _trainStep();
    ++_steps_done;

    if (_steps_done % _target_update_freq == 0) {
        _updateTargetNetwork();
    }
}


std::vector<int> DQNAgent::train(unsigned numEpisode) {
    this->_env->setDebug(false);
    std::vector<int> rewards;
    
    auto pb = ProgressBar("Training", numEpisode, [this, &rewards](const unsigned episode) {
        std::vector<unsigned> s = this->_env->reset();
        bool done = false;
        
        int episodeRewards = 0;

        while (!done) {
            unsigned a = getBehaviorPolicy(s, _steps_done);
            int r;
            std::vector<unsigned> sPrime;
            std::tie(sPrime, r, done) = this->_env->step(a);

            this->update(s, a, r, sPrime); // ++_steps_done is done here

            episodeRewards += r;
            s = sPrime;
        }

        rewards.push_back(episodeRewards);
    });

    return rewards;
}

void DQNAgent::rollout() {
    std::vector<unsigned> s = this->_env->reset();
    this->_env->setDebug(true);
    bool done = false;
    
    unsigned t = 0;
    while (!done) {
        int r = 0;
        unsigned a = getTargetPolicy(s);
        std::vector<unsigned> sPrime;
        std::tie(sPrime, r, done) = this->_env->step(a);

        s = sPrime;
        ++t;
    }
    std::cout << "Took " << t << " time steps to finish!" << std::endl;
}

unsigned DQNAgent::_argmax(const torch::Tensor& v) {
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

/**
 * @brief Performs one training step of the DQN using a minibatch from the replay buffer.
 *
 * This method implements the core learning algorithm of DQN:
 * - Samples a minibatch of transitions from the replay buffer.
 * - Computes Q(s, a) from the current Q-network.
 * - Computes target Q-values using the target network and the Bellman equation.
 * - Calculates the mean squared error (MSE) loss.
 * - Performs a gradient descent step to minimize the loss.
 *
 * If there are not enough samples in the replay buffer, this function exits early.
 */
void DQNAgent::_trainStep() {
    // Not enough samples to train...yet
    if (_replay_buffer.size() < _batch_size) {
        return;
    }

    // Sample a minibatch of transitions
    // # batch = memory.sample(batch_size)
    auto batch = _replay_buffer.sample(_batch_size);

    // # loss = train_dqn_batch() starts here #

    // PRE-processing: Extract data from "batch" tuple
    std::vector<std::vector<unsigned>> states, next_states;
    std::vector<float> rewards;
    std::vector<unsigned> actions;
    std::vector<bool> dones;

    for (const auto& transition : batch) {
        states.push_back(transition.state);
        actions.push_back(transition.action);
        rewards.push_back(transition.reward);
        next_states.push_back(transition.next_state);
        dones.push_back(transition.done);
    }

    // Convert to tensors
    torch::Tensor states_tensor = torch::tensor(states, torch::kFloat32);
    torch::Tensor actions_tensor = torch::tensor(actions, torch::kLong);
    torch::Tensor rewards_tensor = torch::tensor(rewards, torch::kFloat32);
    torch::Tensor next_states_tensor = torch::tensor(next_states, torch::kFloat32);
    torch::Tensor dones_tensor = torch::tensor(dones, torch::kBool);

    // # dqn_model(states).gather(1, actions) # Compute values
    torch::Tensor q_values = _q_net->forward(states_tensor);
    torch::Tensor q_selected = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1);

    // Compute TD target Q-values: r + γ * max_a' Q_target(s', a') using target network
    torch::Tensor next_q_values = _target_net->forward(next_states_tensor);
    torch::Tensor max_next_q_values = std::get<0>(next_q_values.max(1));
    torch::Tensor target = rewards_tensor + _gamma * max_next_q_values * (~dones_tensor);

    // Compute mean squared error loss, like ex6's architecture
    torch::Tensor loss = torch::mse_loss(q_selected, target.detach());

    // Backward pass
    _optimizer.zero_grad();
    loss.backward();
    _optimizer.step();

    // #TODO losses.append(loss)
}

/**
 * Updates the target network by copying weights from the Q-network.
 */
void DQNAgent::_updateTargetNetwork() {
    // Note: state_dict() returns OrderedDict that contains copies, not references.
    this->_target_net->load_state_dict(this->_q_net->state_dict());
}