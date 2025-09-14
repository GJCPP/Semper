#pragma once

#include <format>
#include <random>
#include <unordered_map>

#include "header"

#include "CNN.h"
#include "ligero.h"
#include "goldilocks_quadratic_ext.h"
#include "data_loader.h"

class VGG11 : public CNN {
public:
    VGG11(std::string data_dir, int epoch, int64_t scale, int64_t max_value, uint64_t rho_inv, uint64_t sec_param);
};



// static std::unordered_map<size_t, size_t> relu_map;
