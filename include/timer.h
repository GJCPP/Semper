#pragma once
#include <iostream>
#include <chrono>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <mutex>

#define PRINT_INFO false

class Timer {
public:
    static Timer& getInstance() {
        static Timer instance;
        return instance;
    }

    void start(const std::string& label) {
        mut.lock();
        if (ind.count(label)) {
            mut.unlock();
            resume(label);
            return;
        }
        ind[label] = nextInd++;
        timers.push_back(TimerData{Clock::now(), 0, 1});
        labels.push_back(label);
        mut.unlock();
    }

    void pause(const std::string& label) {
        std::lock_guard<std::mutex> lock(mut);
        auto id = ind.find(label);
        if (id == ind.end()) {
            std::cerr << "pause failed, label not found: " << label << std::endl;
            return;
        }
        if (timers[id->second].req == 0) {
            std::cerr << "Timer '" << label << "' is already paused" << std::endl;
            return;
        }
        auto now = Clock::now();
        timers[id->second].accumulated += std::chrono::duration_cast<std::chrono::milliseconds>(now - timers[id->second].startTime).count();
        --timers[id->second].req;
    }

    void resume(const std::string& label) {
        std::lock_guard<std::mutex> lock(mut);
        auto id = ind.find(label);
        if (id == ind.end()) {
            std::cerr << "resume failed, label not found: " << label << std::endl;
            return;
        }
        // if (timers[id->second].paused > 0) {
        //     std::cerr << "Timer '" << label << "' is not paused" << std::endl;
        //     return;
        // }
        timers[id->second].startTime = Clock::now();
        ++timers[id->second].req;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mut);
        nextInd = 0;
        ind.clear();
        timers.clear();
        labels.clear();
    }

    void print(const std::string& label) const {
        auto id = ind.find(label);
        if (id == ind.end()) {
            std::cerr << "print failed, label not found: " << label << std::endl;
            return;
        }
        std::cout << "[" << label << "] cost: " << double(timers[id->second].accumulated) / 1000 << " s." << std::endl;
    }

    void remove(const std::string& label) {
        std::lock_guard<std::mutex> lock(mut);
        auto id = ind.find(label);
        if (id == ind.end()) {
            std::cerr << "remove failed, label not found: " << label << std::endl;
            return;
        }
        ind.erase(id);
    }

    void printAll() {
        std::lock_guard<std::mutex> lock(mut);
        for (int i = 0; i != nextInd; ++i) {
            if (timers[i].req > 0) {
                std::cout << "Warning: Timer '" << labels[i] << "' is still running." << std::endl;
            }
            std::cout << "[" << labels[i] << "] cost: " << double(timers[i].accumulated) / 1000 << " s." << std::endl;
        }
    }

private:
    std::mutex mut;
    using Clock = std::chrono::high_resolution_clock;

    struct TimerData {
        Clock::time_point startTime;
        uint64_t accumulated;
        int req;
    };

    int nextInd = 0;
    std::vector<TimerData> timers;
    std::vector<std::string> labels;
    std::unordered_map<std::string, int> ind;

    Timer() = default;
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;
};

inline void pause_timer(const std::string& label) {
    Timer::getInstance().pause(label);
}

inline void resume_timer(const std::string& label) {
    Timer::getInstance().resume(label);
}

inline void set_timer(const std::string& label) {
    Timer::getInstance().start(label);
}

// inline void end_timer(const std::string& label) {
//     Timer::getInstance().remove(label);
// }

inline void clear_all_timers() {
    Timer::getInstance().clear();
}

inline void print_all_timers() {
    Timer::getInstance().printAll();
}

class startTimer {
public:
    startTimer(const std::string& label) : label(label) {
        Timer::getInstance().start(label);
    }
    ~startTimer() {
        Timer::getInstance().pause(label);
    }
protected:
    std::string label;
};