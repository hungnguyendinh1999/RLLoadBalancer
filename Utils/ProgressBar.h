#ifndef PROGRESSBAR_H
#define PROGRESSBAR_H
#include <chrono>
#include <functional>
#include <string>

class ProgressBar {
public:
    explicit ProgressBar(const std::string &label, unsigned total, const std::function<void(unsigned it)> &execFunc, unsigned width = 50);
private:
    std::string _label;
    unsigned _total;
    unsigned _width;
    unsigned _progress;
    std::chrono::steady_clock::time_point _lastTime;
    long long _totalTime;

    void _tick();
};

#endif //PROGRESSBAR_H
