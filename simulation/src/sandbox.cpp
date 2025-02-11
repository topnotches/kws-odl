#include <iostream>
#include <fstream>
#include <stdio.h>
#include <assert.h>
#include <sstream>
#include <vector>
#include <numeric>
#include <math.h>
#include "dataloader.hpp"
#include "misc_utils.hpp"
#include "defs.hpp"
#include "layer.hpp"
#include "model_utils.hpp"
#include "layer_analyzer.hpp"

extern layer_analyzer batchnorm_analyzer;
extern layer_analyzer conv_analyzer;
extern layer_analyzer crossloss_analyzer;
extern layer_analyzer dense_fw_analyzer;
extern layer_analyzer dense_bw_analyzer;
extern layer_analyzer dw_analyzer;
extern layer_analyzer fusion_fw_analyzer;
extern layer_analyzer fusion_bw_analyzer;
extern layer_analyzer relu_analyzer;
extern layer_analyzer softmax_fw_analyzer;
extern layer_analyzer softmax_bw_analyzer;


float average(const float* vec, uint16_t size) {
    float sum = 0.0f;
    for (uint16_t i = 0; i < size; i++) {
        sum += vec[i];
    }
    return sum / size;
}

auto load_batch_from_paths(std::vector<std::string> paths) {
    std::vector<float> input_mfccs = {};
    for (auto path : paths) {
        auto temp_floats = load_mffcs_bin(path);
        input_mfccs.insert(input_mfccs.end(), temp_floats.begin(), temp_floats.end());
    }
    return input_mfccs;
}

int  main() {
    uint32_t run = 0;
    while (run < TOTAL_RUNS) {
        auto validation_errors_file_name = "./val_loss_logs/run_"+ std::to_string(run) +".csv";
        run++;
        std::string exported_params = "./exported_models/model_baseline_hyperrparams_12_class.csv";

        std::vector<std::string> test_mfccs_path = {"../dataset_mfccs_raw/yes/d21fd169_nohash_0","../dataset_mfccs_raw/yes/d21fd169_nohash_0","../dataset_mfccs_raw/yes/d21fd169_nohash_1"};
        
        std::vector<float> input_mfccs = load_batch_from_paths(test_mfccs_path);
        
        std::vector<std::string> words = {"yes","no","up","down","left","right","on","off","stop","go"};
        
        std::vector<layer> model = get_model(exported_params, BATCH_SIZE, NUMBER_OF_CLASSES);

        layer softmax(LayerTypes::softmax, model.back().get_output_size());
        layer crossentropy(LayerTypes::cross_entropy_loss, softmax.get_output_size());
#if DO_LAYER_ANALYSIS
        dataloader dataloader(words, "c50f55b8_nohash_19", BATCH_SIZE, 1); // Change '/' to the userID
#else
        dataloader dataloader(words, "c50f55b8", BATCH_SIZE, 0.7); // Change '/' to the userID
#endif
        float error = 0.0f;
        float momentum = 0.0;
        uint32_t i = 0;
        uint32_t ii = 0;
        float err_highest = 0.0f;
        float err_initial = 0.0f;
        float err_lowest = 1000000;
        int epoch_err_highest = 0.0f;
        int epoch_err_lowest = 0.0f;
        
        std::vector<float> all_validation_error;
        std::vector<float> all_validation_accuracies;
        
        while (i < EPOCHS) {
            
            if (dataloader.get_training_pool_empty()) {
                ii++;
                auto mybatch = dataloader.get_batch();
                std::vector<float> labels_onehot;
                // Convert int labels to vector of dim [batchsize classes]
                for (auto label : std::get<1>(mybatch)) {
                    std::vector<float> temp;
                    temp = int_to_float_onehot(2 + label,12);

                    labels_onehot.insert(labels_onehot.end(), temp.begin(), temp.end());
                }
                model_forward(model, std::get<0>(mybatch));
                softmax.forward(model.back().layer_outputs.data());
                crossentropy.forward(softmax.layer_outputs.data(), labels_onehot.data());
                softmax.backward(labels_onehot.data());
                model[29].backward(softmax.layer_gradient_outputs.data()); //dense
                model[28].backward(model[29].layer_gradient_outputs.data()); //fusion
                float temp_err = 0.0f;

                //std::cout << softmax.get_output_size().width << std::endl;
                
                for (auto output : crossentropy.layer_outputs) {
                    temp_err += output;
                }
                temp_err /= crossentropy.layer_outputs.size()/BATCH_SIZE;

                //for (auto bobber : model[28].layer_gradient_outputs) {
                //    bobberlobber += abs(bobber);
                //}

                //float sum_of_elems = 0.0f;
                //for(std::vector<float>::iterator it = model[LAYER_SELECT].layer_outputs.begin(); it != model[LAYER_SELECT].layer_outputs.end(); ++it)
                //sum_of_elems += *it;
                //std::cout << sum_of_elems << std::endl;

#if DO_LAYER_ANALYSIS
                batchnorm_analyzer.print_stats_colnames();
                batchnorm_analyzer.print_stats_raw();
                conv_analyzer.print_stats_raw();
                crossloss_analyzer.print_stats_raw();
                dense_fw_analyzer.print_stats_raw();
                dense_bw_analyzer.print_stats_raw();
                dw_analyzer.print_stats_raw();
                fusion_fw_analyzer.print_stats_raw();
                fusion_bw_analyzer.print_stats_raw();
                relu_analyzer.print_stats_raw();
                softmax_fw_analyzer.print_stats_raw();
                softmax_bw_analyzer.print_stats_raw();
#else
                //dataloader.print_progress_bar(i+1,error);

#endif

            } else {
                momentum = 0.0;
                auto myvalset = dataloader.get_validation_set();
                float temp_err = 0.0f;
                float total = 0;
                float correct = 0;
                //std::cout <<bobberlobber << std::endl;
                //std::cout <<bobberlobber << std::endl;
                // Convert int labels to vector of dim [batchsize classes]
                
                // Good luck reading this shit
                for (uint val_index = 0; val_index < dataloader.get_validation_size()/BATCH_SIZE; val_index++) {
                    // Dummy read for compiler e
                    auto vlabel = std::get<1>(myvalset)[0];
                    auto vinput = std::get<0>(myvalset)[0];
                    std::vector<float> labels_onehot;
                    std::vector<float> actual_inputs;
                    for (uint val_index_in_batch = 0; val_index_in_batch < BATCH_SIZE; val_index_in_batch++) {
                        vlabel = std::get<1>(myvalset)[val_index * BATCH_SIZE + val_index_in_batch];
                        vinput = std::get<0>(myvalset)[val_index * BATCH_SIZE + val_index_in_batch];
                        std::vector<float> temp;
                        temp = int_to_float_onehot(2 + vlabel,12);
                        labels_onehot.insert(labels_onehot.end(), temp.begin(), temp.end());
                        actual_inputs.insert(actual_inputs.end(), vinput.begin(), vinput.end());
                    }

                    model_forward(model, actual_inputs);
                    softmax.forward(model.back().layer_outputs.data());
                    crossentropy.forward(softmax.layer_outputs.data(), labels_onehot.data());
                    
                    for (auto output : crossentropy.layer_outputs) {
                        temp_err += output;
                    }
                    
                    // Calculate predictions (softmax output)
                    for (uint i = 0; i < BATCH_SIZE; i++) {
                        float max_output = -1.0f;
                        int predicted_label = -1;
                        // Find the predicted class (argmax of softmax outputs)
                        for (int j = 0; j < 12; j++) {
                            if (softmax.layer_outputs[i * 12 + j] > max_output) {
                                max_output = softmax.layer_outputs[i * 12 + j];
                                predicted_label = j;
                            }
                        }

                        // Check if prediction matches label
                        if (predicted_label == 2 + vlabel) {
                            correct = correct + 1.0f;
                        }
                        total = total + 1.0f;
                    }
                }
                
                /*
                for (auto label : std::get<1>(myvalset)) {
                    std::vector<float> temp;
                    temp = int_to_float_onehot(2 + label,12);
                    
                    labels_onehot.insert(labels_onehot.end(), temp.begin(), temp.end());
                }
                model_forward(model, std::get<0>(myvalset));
                softmax.forward(model.back().layer_outputs.data());
                crossentropy.forward(softmax.layer_outputs.data(), labels_onehot.data());
                
                for (auto output : crossentropy.layer_outputs) {
                    temp_err += output;
                }
                */
                temp_err /= dataloader.get_validation_size();
                error = (momentum)*error+(1-momentum)*temp_err;
                if (error > err_highest) {
                    err_highest       = error;
                    epoch_err_highest = i;
                }
                if (error < err_lowest) {
                    err_lowest       = error;
                    epoch_err_lowest = i;
                }
                if (i == 0) {
                    err_initial       = error;
                }
                all_validation_error.push_back(error);
                all_validation_accuracies.push_back(correct/total);
                dataloader.reset_training_pool();
                //std::cout << std::endl; // New line between epochs
                i++;
            }
        }
        /*
        std::cout << "END OF ONLINE TRAINING" << std::endl;
        std::cout << "INITIAL ERROR: " << err_initial << std::endl;
        std::cout << "HIGHEST ERROR: " << err_highest << "  AT EPOCH:" << epoch_err_highest << std::endl;
        std::cout << "LOWEST ERROR: " << err_lowest << "  AT EPOCH:" << epoch_err_lowest << std::endl;
        std::cout << "FINAL ERROR: " << error << std::endl;
        */


        // Open CSV file
        std::ofstream file(validation_errors_file_name);
        
        if (!file.is_open()) {
            std::cerr << "Error opening file!" << std::endl;
            return 1;
        }

        // Write CSV header
        file << "Epoch,Val_Loss,Val_Acc" << std::endl;

        // Write error messages and codes
        for (size_t i = 0; i < all_validation_error.size(); ++i) {
            file << i << "," << all_validation_error[i] << "," << all_validation_accuracies[i] << std::endl;
        }

        // Close file
        file.close();
    }

    return 0;
}
