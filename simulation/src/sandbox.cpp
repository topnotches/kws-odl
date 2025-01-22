#include <iostream>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <vector>
#include <math.h>
#include <dataloader.hpp>
#include "misc_utils.hpp"
#include "defs.hpp"
#include "layer.hpp"
#include "model_utils.hpp"


auto load_batch_from_paths(std::vector<std::string> paths) {
    std::vector<float> input_mfccs = {};
    for (auto path : paths) {
        auto temp_floats = load_mffcs_bin(path);
        input_mfccs.insert(input_mfccs.end(), temp_floats.begin(), temp_floats.end());
    }
    return input_mfccs;
}

int  main() {

    std::string exported_params = "./exported_models/export_params_nclass_12.csv";

    std::vector<std::string> test_mfccs_path = {"../dataset_mfccs_raw/yes/d21fd169_nohash_0","../dataset_mfccs_raw/yes/d21fd169_nohash_0","../dataset_mfccs_raw/yes/d21fd169_nohash_1"};
    
    std::vector<float> input_mfccs = load_batch_from_paths(test_mfccs_path);
    
    std::vector<std::string> words = {"yes"};
    
    std::vector<layer> model = get_model(exported_params, BATCH_SIZE, NUMBER_OF_CLASSES);
    dataloader dataloader(words, "yes", BATCH_SIZE);
    // Model is list of layers
    uint8_t i = 0;
    while (i < EPOCHS) {
        if (dataloader.get_training_pool_empty()) {
            auto mybatch = dataloader.get_batch();
            
            model_forward(model, std::get<0>(mybatch));

            dataloader.print_progress_bar(i);
        } else {
            dataloader.reset_training_pool();
            std::cout << "EPOCH: " << std::endl;
            i++;
        }
    }
    return 0;
}