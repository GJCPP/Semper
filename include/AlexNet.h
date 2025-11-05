#pragma once

#include <format>
#include <random>
#include <unordered_map>

#include "header"

#include "CNN.h"
#include "ligero.h"
#include "goldilocks_quadratic_ext.h"
#include "data_loader.h"

class AlexNet : public CNN {
public:
    AlexNet(std::string data_dir, int epoch, int64_t scale, int64_t max_value, uint64_t rho_inv, uint64_t sec_param);
};

