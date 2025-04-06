#include "ProgressBar.h"

#include <iomanip>
#include <iostream>

ProgressBar::ProgressBar(const std::string &label, const unsigned total,
                            const std::function<void(unsigned it)> &execFunc, const unsigned width) {
    this->_label = label;
    this->_progress = 0;
    this->_total = total;
    this->_width = width;
    this->_totalTime = 0;

    for (unsigned i = 0; i < this->_total; i++) {
        execFunc(i);
        this->_tick();
    }
}


void ProgressBar::_tick() {
    this->_progress += 1;
    if (this->_progress == 1) {
        this->_lastTime = std::chrono::steady_clock::now();;
    } else {
        const auto newTime = std::chrono::steady_clock::now();
        this->_totalTime += std::chrono::duration_cast<std::chrono::milliseconds>(
            newTime - this->_lastTime).count();
        this->_lastTime = newTime;
    }

    float iterationsPerSec = 0;
    if (this->_totalTime > 1) {
        iterationsPerSec = (1000.0f * this->_progress) / this->_totalTime;
    }

    const float remainingTimeSecs = (this->_total - this->_progress) / iterationsPerSec;

    const int hours = static_cast<int>(remainingTimeSecs) / 3600;
    const int minutes = (static_cast<int>(remainingTimeSecs) % 3600) / 60;
    const int seconds = static_cast<int>(remainingTimeSecs) % 60;

    const float ratio = static_cast<float>(this->_progress) / this->_total;
    const int filled = this->_width * ratio;

    std::cout << "\r" << this->_label << ": " << static_cast<int>(ratio * 100.0) << "% " << "[";
    for (int i = 0; i < this->_width; ++i) {
        if (i < filled)
            std::cout << "=";
        else
            std::cout << " ";
    }
    std::cout << "] (" << this->_progress << "/" << this->_total << ") ["
                << std::setw(2) << std::setfill('0') << hours << ":"
                << std::setw(2) << std::setfill('0') << minutes << ":"
                << std::setw(2) << std::setfill('0') << seconds
                << ", " << iterationsPerSec << "it/s]";
    std::cout.flush();

    if (this->_progress == this->_total) {
        std::cout << std::endl;
    }
}
