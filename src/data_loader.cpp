#include "data_loader.h"

cnpy::npz_t loadDataset(const std::string& dir) {
    std::string data_file = dir + "/dataset.npz";
    return cnpy::npz_load(data_file);
}

cnpy::npz_t loadEpochData(const std::string& dir, int epoch) {
    std::string data_file = dir + "/epoch_" + std::to_string(epoch) + ".npz";
    return cnpy::npz_load(data_file);
}

cnpy::npz_t loadConvWitForward(const std::string& dir, int epoch, int batch, int conv_id) {
    std::string data_file = std::format("{}/epoch_{}_witness/batch_{}/conv_{}_forward.npz", dir, epoch, batch, conv_id);
    return cnpy::npz_load(data_file);
}



