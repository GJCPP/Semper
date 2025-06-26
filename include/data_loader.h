#pragma once
#include <string>
#include <vector>
#include <format>
#include <random>
#include <cnpy.h>

#include "highdim_array.h"

cnpy::npz_t loadEpochData(const std::string& dir, int epoch);
