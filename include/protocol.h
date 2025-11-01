#pragma once

#include <stdint.h>

#include <vector>
#include <memory>
#include <mutex>

#include "timer.h"

class protocol {
public:
    protocol(uint64_t _sec_param) : sec_param(_sec_param) {}
    virtual bool execute() = 0;
    virtual ~protocol() = default;
protected:
    uint64_t sec_param;
};

class protoque {
public:
    void push_back(std::unique_ptr<protocol>&& p) {
        std::lock_guard<std::mutex> lock(mut);
        protocols.push_back(std::move(p));
    }

    bool execute_all() {
        // std::lock_guard<std::mutex> lock(mut);
        startTimer timer("prove protoque");
        bool ret = true;

        for (auto& p : protocols) {
            if (!p->execute()) ret = false;
        }

        protocols.clear();
        return ret;
    }

protected:
    std::vector<std::unique_ptr<protocol>> protocols;
    std::mutex mut;
};
