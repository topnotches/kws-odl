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
    
    std::vector<std::string> words = {"yes","no"};
    
    std::vector<layer> model = get_model(exported_params, BATCH_SIZE, NUMBER_OF_CLASSES);

    layer softmax(LayerTypes::softmax, model.back().get_output_size());
    layer crossentropy(LayerTypes::cross_entropy_loss, softmax.get_output_size());

    dataloader dataloader(words, "/", BATCH_SIZE);

    float error = 0.0f;
    float momentum = 0.01;
    uint8_t i = 0;
    while (i < EPOCHS) {
        if (dataloader.get_training_pool_empty()) {
            auto mybatch = dataloader.get_batch();
            std::vector<float> labels_onehot;
            // Convert int labels to vector of dim [batchsize classes]
            for (auto label : std::get<1>(mybatch)) {
                std::vector<float> temp;
                if (label == 0) {
                    temp = int_to_float_onehot(2,12);
                } else if (label == 1) {
                    temp = int_to_float_onehot(3,12);
                }
                labels_onehot.insert(labels_onehot.end(), temp.begin(), temp.end());
            }
            model_forward(model, std::get<0>(mybatch));
            softmax.forward(model.back().layer_outputs.data());
            
            crossentropy.forward(softmax.layer_outputs.data(), labels_onehot.data());
            float temp_err = 0.0f;
            for (auto output : crossentropy.layer_outputs) {
                temp_err += output;
            }
            temp_err /= crossentropy.layer_outputs.size();

            error = error*(momentum) + temp_err*(1-momentum);
            dataloader.print_progress_bar(i,error);
        } else {
            dataloader.reset_training_pool();
            std::cout << std::endl; // New line between epochs
            i++;
        }
    }
    return 0;
}