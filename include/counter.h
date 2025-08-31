#pragma once

#include <string>
#include <map>
#include <set>
#include <iostream>

// count proof size
class Counter {
public:
    enum metric {
        B, KB, MB
    };
    Counter() = default;

    void add(const std::string& str, size_t sz) {
        if (counts.find(str) == counts.end()) {
            counts[str] = 0;
        }
        counts[str] += sz;
    }

    void add(size_t sz) {
        for (auto& s : active) {
            add(s, sz);
        }
    }

    void start(const std::string& str) {
        active.insert(str);
    }

    void end(const std::string& str) {
        active.erase(str);
    }

    void clear() {
        counts.clear();
        active.clear();
    }

    void print(const std::string& name, metric m) const {
        double scale_val = scale.at(m);
        std::string unit_val = unit.at(m);
        std::cout << name << ": " << double(counts.at(name)) / scale_val << unit_val << std::endl;
    }

    void print_all(metric m) {
        for (const auto& [key, value] : counts) {
            print(key, m);
        }
    }

protected:
    std::map<std::string, size_t> counts;
    std::set<std::string> active;
    const static std::map<metric, double> scale;
    const static std::map<metric, std::string> unit;
};



void start_proof(const std::string& name);

void end_proof(const std::string& name);

void clear_proof();

void add_proof_size(size_t sz);

void print_proof_size(const std::string& str, Counter::metric m);

void print_all_proof_size(Counter::metric m);


class startCounter {
public:
    startCounter(const std::string& name) 
        : proof_name(name) {
        start_proof(name);
    }

    ~startCounter() {
        end_proof(proof_name);
    }
protected:
    std::string proof_name;
};

