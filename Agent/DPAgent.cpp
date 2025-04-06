#include "DPAgent.h"

#include <algorithm>
#include <sstream>

/**
 * 
 */
DPAgent::DPAgent(const std::shared_ptr<Environment> &environment, double gamma, double theta) 
    : environment(environment), gamma(gamma), theta(theta) {}

/**
 * Convert input state matrix to V-table string key
 */
std::string DPAgent::state_to_key(const std::vector<unsigned>& s) const {
    std::stringstream ss;
    for (unsigned v : s) {
        ss << v << ",";
    }
    return ss.str();
}

void DPAgent::run_value_iteration() {
    unsigned num_actions = this->environment->getNumAction();
    unsigned max_task = this->environment->getNumTask();
    unsigned max_thread = this->environment->getMaxThread();
    unsigned num_proc = this->environment->getNumProc();

    // Build all possible states
    std::vector<std::vector<unsigned>> all_states;

    for (unsigned t = 0; t <= max_task; ++t) {
        std::vector<unsigned> init = {t};
        std::vector<std::vector<unsigned>> partial_states = {init};

        for (unsigned p = 0; p < num_proc; ++p) {
            std::vector<std::vector<unsigned>> new_states;
            for (const auto& state : partial_states) {
                for (unsigned bt = 0; bt <= max_thread; ++bt) {
                    std::vector<unsigned> extended = state;
                    extended.push_back(bt);
                    new_states.push_back(extended);
                }
            }
            partial_states = new_states;
        }

        all_states.insert(all_states.end(), partial_states.begin(), partial_states.end());
    }

    // Initialize V(s) = 0
    for (const auto& s : all_states) {
        this->V[state_to_key(s)] = 0.0;
    }

    // Value Iteration Loop
    bool converged = false;
    while (!converged) {
        converged = true;
        for (const auto& s : all_states) {
            std::string skey = state_to_key(s);
            double old_v = this->V[skey];
            double best_v = -std::numeric_limits<double>::infinity();
            unsigned best_a = 0;

            for (unsigned a = 0; a < num_actions; ++a) {
                auto [s_prime, r, done] = this->environment->simulateStep(s, a);
                double v_prime = done ? 0.0 : V[state_to_key(s_prime)];
                double q = r + gamma * v_prime;

                if (q > best_v) {
                    best_v = q;
                    best_a = a;
                }
            }

            this->V[skey] = best_v;
            this->policy[skey] = best_a;

            if (std::abs(best_v - old_v) > theta) {
                converged = false;
            }
        }
    }
}

double DPAgent::get_value(const std::vector<unsigned>& s) const {
    return this->V.at(state_to_key(s));
}

unsigned DPAgent::get_best_action(const std::vector<unsigned>& s) const{
    return this->policy.at(state_to_key(s));
}