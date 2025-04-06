#include "Environment.h"

#include <iostream>
#include <sstream>

Environment::Environment(const unsigned numProc, const unsigned numTask, const unsigned maxThread,
        const unsigned maxDuration, const unsigned seed) {
    this->_isDebug = false;
    this->_processors = std::vector<std::shared_ptr<Processor>>();
    this->_numTask = numTask;
    this->_randomizer = std::mt19937(seed);
    this->_maxThread = maxThread;
    this->_maxDuration = maxDuration;
    this->_numProc = numProc;
    this->_numAction = numProc + 1;

    this->reset();
}

Environment::Environment(const unsigned numProc, const unsigned numTask, const unsigned maxThread,
        const unsigned maxDuration) {
    this->_isDebug = false;
    this->_processors = std::vector<std::shared_ptr<Processor>>();
    this->_numTask = numTask;
    this->_randomizer = std::mt19937(std::random_device()());
    this->_maxThread = maxThread;
    this->_maxDuration = maxDuration;
    this->_numProc = numProc;
    this->_numAction = numProc + 1;

    this->reset();
}

std::vector<unsigned> Environment::reset() {
    std::vector<unsigned> s = {};
    this->_taskQueue.clear();
    this->_processors.clear();

    // Add tasks
    for (int i = 0; i < this->_numTask; ++i) {
        auto rand = std::uniform_int_distribution<unsigned>(1, this->_maxDuration);
        this->_taskQueue.push_back(std::make_shared<Task>(rand(this->_randomizer)));
    }
    s.push_back(this->_numTask);

    // Add processors
    for (int i = 0; i < this->_numProc; ++i) {
        this->_processors.push_back(std::make_shared<Processor>(this->_maxThread));
        s.push_back(0);
    }

    if (this->_isDebug) {
        std::cout << this->toString() << std::endl;
    }

    return s;
}

std::tuple<std::vector<unsigned>, int, bool> Environment::step(const unsigned action) {
    // Action out of bound guard
    if (action >= this->_numAction) {
        std::stringstream es;
        es << "Given action out of range: " << action << std::endl;
        throw std::invalid_argument(es.str());
    }

    // Returning vars: nextState, reward, done
    std::vector<unsigned> nextState;
    int reward = 0;
    bool done = true;

    // Natural tick at the beginning
    for (const std::shared_ptr<Processor>& proc : this->_processors) {
        proc->tick();
    }

    // Perform action
    if (action < this->_numAction - 1) {
        // 0 -> (NumProc - 1): Move from queue to proc, with action being index of the proc to move to
        this->_moveFromQueue(action);
    }
    reward -= 1;

    // Check done
    nextState.push_back(this->_taskQueue.size());
    if (!this->_taskQueue.empty()) {
        done = false;
    }

    for (const std::shared_ptr<Processor>& proc : this->_processors) {
        nextState.push_back(proc->getNumBusyThread());
        if (proc->getNumBusyThread() != 0) {
            done = false;
        }

        if (!this->_taskQueue.empty() && proc->getUtilization() < 0.2) {
            reward -= 1;
        }
    }

    if (this->_isDebug) {
        std::cout << this->toString() << std::endl;
    }

    return std::make_tuple(nextState, reward, done);
}

std::tuple<std::vector<unsigned>, int, bool> Environment::simulateStep(const std::vector<unsigned>& state, unsigned action) {
    // Action out of bound guard
    if (action >= this->_numAction) {
        std::stringstream es;
        es << "Given action out of range: " << action << std::endl;
        throw std::invalid_argument(es.str());
    }

    // Returning vars: nextState, reward, done
    std::vector<unsigned> nextState;
    int reward = 0;
    bool done = true;

    // Pseudo-Processors for simulation
    unsigned num_proc = this->getNumProc();
    unsigned max_threads = this->getMaxThread();
    unsigned num_tasks = state[0];  // task queue size
    std::vector<unsigned> proc_threads(state.begin() + 1, state.end()); // from s[1] to s[4]

    // Simulate tick: decrese one thread per processor
    for (unsigned& threads : proc_threads) {
        // proc->tick();
        if (threads > 0) {
            threads -= 1;
        }
    }

    // Perform action
    if (action < num_proc && num_tasks > 0 && proc_threads[action] < max_threads) {
        proc_threads[action] += 1;
        num_tasks -= 1;
    }

    // Adjust reward
    reward -= 1;

    for (unsigned i = 0; i < num_proc; ++i) {
        float utilization = static_cast<float>(proc_threads[i]) / max_threads;
        if (num_tasks > 0 && utilization < 0.2f) {
            reward -= 10;
        }
    }

    // Check done
    done = (num_tasks == 0);
    for (unsigned threads: proc_threads) {
        if (threads > 0) {
            done = false;
            break;
        }
    }

    if (this->_isDebug) {
        std::cout <<  "haven't implement this yet uhhhhh ohhh" << std::endl;
    }


    nextState.push_back(num_tasks);
    nextState.insert(nextState.end(), proc_threads.begin(), proc_threads.end());

    return std::make_tuple(nextState, reward, done);
}

unsigned Environment::getNumAction() const {
    return this->_numAction;
}

unsigned Environment::getNumTask() const {
    return this->_numTask;
}

unsigned Environment::getMaxThread() const {
    return this->_maxThread;
}

unsigned Environment::getNumProc() const {
    return this->_numProc;
}

std::string Environment::toString() const {
    std::stringstream result;

    result << "===========================================" << std::endl;
    result << "TASK QUEUE: [";
    for (const std::shared_ptr<Task>& task : this->_taskQueue) {
        result << task->getRemainingDuration() << " ";
    }
    result << "]" << std::endl;

    for (int i = 0; i < this->_processors.size(); ++i) {
        result << "PROCESS " << i << ": " << this->_processors.at(i)->getTotalProcessTime() << " STEPS" << std::endl;
    }
    result << "===========================================" << std::endl;

    return result.str();
}

void Environment::setDebug(const bool isDebug) {
    this->_isDebug = isDebug;
}

void Environment::_moveFromQueue(const unsigned toProc) {
    if (this->_taskQueue.empty()) {
        return;
    }

    const std::shared_ptr<Task> t = this->_taskQueue.front();
    this->_taskQueue.pop_front();

    // Processor is at max cap, re-queued back to the task queue
    if (!this->_processors.at(toProc)->queue(t)) {
        this->_taskQueue.push_front(t);
    }
}
