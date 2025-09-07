#include "counter.h"


static Counter proof_size;


const std::map<Counter::metric, double> Counter::scale = {
    {B, 1},
    {KB, 1024},
    {MB, 1024 * 1024}
};

const std::map<Counter::metric, std::string> Counter::unit = {
    {B, "B"},
    {KB, "KB"},
    {MB, "MB"}
};

void start_proof(const std::string& name) {
    proof_size.start(name);
}

void end_proof(const std::string& name) {
    proof_size.end(name);
}

double get_proof_size(const std::string& name, Counter::metric m) {
    return proof_size.get_size(name, m);
}

void clear_proof() {
    proof_size.clear();
}

void add_proof_size(size_t sz) {
    proof_size.add(sz);
}

void print_proof_size(const std::string& str, Counter::metric m) {
    proof_size.print(str, m);
}

void print_all_proof_size(Counter::metric m) {
    proof_size.print_all(m);
}
