#include "data_loader.h"


cnpy::npz_t loadEpochData(const std::string& dir, int epoch) {
    std::string data_file = dir + "/epoch_" + std::to_string(epoch) + ".npz";
    cnpy::npz_t npz_data = cnpy::npz_load(data_file);

    cnpy::npz_t data_map;
    for (auto& pair : npz_data) {
        data_map[pair.first] = pair.second;
    }

    return data_map;
}



