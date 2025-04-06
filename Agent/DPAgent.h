#ifndef DPAGENT_H
#define DPAGENT_H

#include <unordered_map>
#include <vector>
#include <memory>
#include "../Environment/Environment.h"

class DPAgent {
public:
    DPAgent(const std::shared_ptr<Environment> &environment, double gamma, double theta);
    double get_value(const std::vector<unsigned>& s) const;
    unsigned get_best_action(const std::vector<unsigned>& s) const;
    void run_value_iteration();
private:
    std::shared_ptr<Environment> environment;
    double gamma;
    double theta; // convergence threshold

    // key = stringified state, value = V(s)
    std::unordered_map<std::string, double> V;

    // key = stringified state, value = best action
    std::unordered_map<std::string, unsigned> policy;

    unsigned argmax(std::vector<float> v);
    std::string state_to_key(const std::vector<unsigned>& s) const;
};
    
#endif // DPAGENT_H