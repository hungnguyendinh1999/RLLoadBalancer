#include "Processor.h"

Processor::Processor(const unsigned maxThread) {
    this->_maxThread = maxThread;
}

bool Processor::queue(const std::shared_ptr<Task> &task) {
    if (task == nullptr) {
        return false;
    }

    this->_tasks.push_back(task);
    this->_totalProcessTime += task->getRemainingDuration();

    return true;
}

void Processor::tick() {
    unsigned taskToTick = this->getNumBusyThread();
    for (unsigned i = 0; i < taskToTick; i++) {
        // Reduce task's duration. If duration is 0 then task is considered done, remove
        this->_tasks[i]->tick();
        if (this->_tasks[i]->isFinished()) {
            this->_tasks.erase(_tasks.begin() + i);
            --i;
            --taskToTick;
        }

        // Reduce total process time
        if (this->_totalProcessTime != 0) {
            this->_totalProcessTime -= 1;
        }
    }
}

unsigned Processor::getTotalProcessTime() const {
    return this->_totalProcessTime;
}

float Processor::getUtilization() const {
    return static_cast<float>(this->_tasks.size()) / static_cast<float>(this->_maxThread);
}

unsigned Processor::getNumBusyThread() const {
    return std::min(static_cast<unsigned>(this->_tasks.size()), this->_maxThread);
}


