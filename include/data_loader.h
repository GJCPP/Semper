#pragma once
#include <string>
#include <vector>
#include <cnpy.h>

cnpy::npz_t loadEpochData(const std::string& dir, int epoch);

void play_with_data_loader();
