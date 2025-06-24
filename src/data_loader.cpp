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

void play_with_data_loader() {
    cnpy::npz_t data = loadEpochData("/home/gaojc/Desktop/logup-main/training_trace", 0);
    for (auto& [key, value] : data) {
        std::cout << key << " of shape : "<< std::endl;
        for (auto& shape : value.shape) {
            std::cout << shape << " ";
        }
        std::cout << std::endl;
    }
    // Get conv1 weights from the data
    // auto& conv1_weights = data["conv_q1"];
    // std::cout << "\nconv1 weights:" << std::endl;
    
    // Get raw data pointer and cast to int64_t since weights are quantized integers
    // int64_t* weights = conv1_weights.data<int64_t>();
    
    // Print weights based on shape
    // size_t out_channels = conv1_weights.shape[0];
    // size_t in_channels = conv1_weights.shape[1]; 
    // size_t kernel_h = conv1_weights.shape[2];
    // size_t kernel_w = conv1_weights.shape[3];
    
    // for(size_t oc = 0; oc < out_channels; oc++) {
    //     std::cout << "Output channel " << oc << ":\n";
    //     for(size_t ic = 0; ic < in_channels; ic++) {
    //         std::cout << "Input channel " << ic << ":\n";
    //         for(size_t h = 0; h < kernel_h; h++) {
    //             for(size_t w = 0; w < kernel_w; w++) {
    //                 size_t idx = oc * (in_channels * kernel_h * kernel_w) + 
    //                             ic * (kernel_h * kernel_w) +
    //                             h * kernel_w + w;
    //                 std::cout << weights[idx] << " ";
    //             }
    //             std::cout << "\n";
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << "\n";
    // }
}
