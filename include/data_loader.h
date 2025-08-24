#pragma once
#include <string>
#include <vector>
#include <format>
#include <random>

#include "cnpy.h"

#include "array_view.h"

cnpy::npz_t loadDataset(const std::string& dir);

cnpy::npz_t loadEpochData(const std::string& dir, int epoch);

cnpy::npz_t loadConvWitForward(const std::string& dir, int epoch, int batch, int conv_id);

cnpy::npz_t loadConvWitDW(const std::string& dir, int epoch, int batch, int conv_id);

cnpy::npz_t loadConvWitDX(const std::string& dir, int epoch, int batch, int conv_id);
