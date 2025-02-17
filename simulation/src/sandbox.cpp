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
    std::vector<std::string> all_user_ids = {"c50f55b8", "050170cb", "964e8cfd", "bbd0bbd0", "7846fd85", "b528edb3", "07089da9", "cd671b5f", "5ebc1cda", "7213ed54", "92a9c5e6", "8fe67225", "c1d39ce8", "d962e5ac", "28ed6bc9", "b0ae6326", "b8874962", "f2f0d244", "a1e71565", "3a3ee7ed", "ddedba85", "56eb74ae", "7ea032f3", "c8b904e3", "2bdbe5f7", "b97c9f77", "3cc595de", "3b3d2f59", "69a1a79f", "eb3f7d82", "7e7ca854", "9b8a7439", "bde0f20a", "0d90d8e1", "b5cf6ea8", "cd85758f", "97f4c236", "645ed69d", "679463fc", "95fd5a0f", "067f61e2", "617de221", "20fd451f", "f68160c0", "1ecfb537", "c9b653a0", "0e6e36c9", "6565a81d", "3d794813", "fb7eb481", "d5b963aa", "b29f8b23", "31f01a8d", "9151f184", "1496195a", "9ff2d2f4", "fb727898", "d9aa8c90", "ca4d5368", "ec989d6d", "893705bb", "234d6a48", "4e6902d0", "3c257192", "fbb56351", "e3e0f145", "da040491", "72242187", "b9515bf3", "8eb4a1bf", "f30285c8", "8281a2a8", "f9643d42", "f2e9b610", "b87bdb22", "a05a90c1", "18e910f4", "0ba018fc", "b4aa9fef", "cd7f8c1b", "d070ea86", "d3831f6a", "0d82fd99", "96a48d28", "322d17d3", "122c5aa7", "321aba74", "ace072ba", "a9ca1818", "4c6944d6", "3bfd30e6", "d21fd169", "779de043", "35d1b6ee", "834f03fe", "525eaa62", "f822b9bf", "02ade946", "72e382bd", "332d33b1", "94d370bf", "845f8553", "f736ab63", "b0f5b16d", "85851131", "b66f4f93", "8134f43f", "9aa21fa9", "8e05039f", "a7200079", "686d030b", "a5d1becc", "acfd5b5e", "b49caed3", "0ff728b5", "eb0676ec", "d278d8ef", "324210dd", "b2ae3928", "e6327279", "61e2f74f", "b4ea0d9a", "37dca74f", "38d78313", "30060aba", "e71a9381", "ed032775", "d394ef8e", "fb8c31a9", "f5341341", "9a76f8c3", "179a61b7", "0717b9f6", "28e47b1a", "5c39594f", "ac9dee0e", "c22ebf46", "dae01802", "c1e0e8e3", "195c120a", "742d6431", "f568162b", "439c84f4", "616420be", "c79159aa", "1b4c9b89", "15dd287d", "bcf614a2", "a929f9b9", "ffd2ba2f", "131e738d", "c9b5ff26", "2aca1e72", "bb6d4301", "57152045", "8dc18a75", "953fe1ad", "f5626af6", "978240e1", "4cee0c60", "a6d586b7", "25132942", "c120e80e", "3bb68054", "1e02ffc5", "d1dd03ed", "fc3ba625", "c93d5e22", "9be15e93", "29fb33da", "b5552931", "14c7b073", "51c5601d", "4fd1443e", "3a69f765", "2927c601", "15c563d7", "e53139ad", "9886d8bf", "833d9f56", "1a994c9f", "c39703ec", "81332c92", "c22d3f18", "d264f7b6", "211b928a", "6c968bd9", "5f5eef60", "e41a903b", "1e9e6bdd", "229978fd", "1dc86f91", "5cf1ecce", "b69002d4", "21832144", "de89e2ca", "61abbf52", "f5733968", "08ab231c", "a77fbcfd", "890cc926", "2e75d37a", "3102f006", "2fcb6397", "94de6a6a", "32ad5b65", "ef77b778", "cb72dfb6", "333784b7", "98ea0818", "333d7ddb", "ad63d93c", "73f20b00", "87d5e978", "c7dc7278", "6ace4fe1", "64df20d8", "c634a189", "2f0ce4d9", "89947bd7", "e0315cf6", "0ea9c8ce", "3c4aa5ef", "ca4912b6", "f2e59fea", "cfbedff9", "f19c1390", "b9f46737", "113b3fbc", "e1469561", "fa57ab3b", "3f45a0cf", "7211390b", "0f46028a", "87070229", "4c3cddb8", "f01a0d63", "beb458a4", "3d53244b", "0b7ee1a0", "106a6183", "e11fbc6e", "bab36420", "ef3367d9", "bdee441c", "676f8138", "017c4098", "bbc30633", "6ef407da", "1ed0b13d", "3291330e", "71d0ded4", "b308773d", "e9bc5cc2", "cc6ee39b", "6727b579", "63f7a489", "9e92ef0c", "9b02d503", "3e31dffe", "8769c34c", "c948d727", "70a00e98", "0137b3f4", "f0659908", "8b25410a", "574ec766", "763188c4", "bdd22e4d", "a7acbbeb", "499be02e", "2da58b32", "472b8045", "db24628d", "ff2b842e", "888a0c49", "a16013b7", "5170b77f", "235b444f", "b83c1acf", "0f7266cf", "3589bc72", "171edea9", "a2cc5444", "365908bd", "0d6d7360", "5b09db89", "fbf3dd31", "aff582a1", "06f6c194", "6aa8def4", "25e95412", "5f9cd2eb", "0eb48e10", "8e523821", "ce0cb033", "b69fe0e2", "2b5e346d", "74241b28", "3bc21161", "0132a06d", "692a88e6", "ccca5655", "f798ac78", "e4be0cf6", "e7ea8b76", "42a99aec", "fa52ddf6", "3b195250", "a04817c2", "42f81601", "29229c21", "4a1e736b", "c6ee87a7", "2356b88d", "b76f6088", "51055bda", "23da904f", "471a0925", "3a929277", "30276d03", "d98dd124", "10ace7eb", "a045368c", "c4e00ee9", "eaa83485", "24befdb3", "f035e2ea", "201e28a9", "f8ba7c0e", "773e26f7", "f0522ff4", "bfd26d6b", "39833acb", "099d52ad", "1acc97de", "f575faf3", "3b4f8f24", "6c429c7b", "54aecbd5", "ee1d8d49", "eeaf97c3", "2fee065a", "a1cff772", "51f7a034", "7fb8d703", "6bf5baf3", "c0445658", "54b6d355", "da584bc0", "617aeb6c", "61ab8fbc", "197f4153", "e49428d9", "cce7416f", "578d3efb", "bd76a7fd", "cc4f9250", "5a5721f8", "d0faf7e4", "ceef6d96", "e0c782d5", "e882abb2", "cdee383b", "bb05582b", "bbf38549", "22296dbe", "5e3dde6b", "cb62dbf1"};
    #pragma omp parallel for num_threads(18) 
    for (auto uid : all_user_ids) {
        for (uint32_t run = 0; run < TOTAL_RUNS; run++) {
            auto validation_errors_file_name = "./user_perf_logs/"+uid+"_run_"+ std::to_string(run) +".csv";
            std::string exported_params = "./exported_models/export_params_nclass_10.csv";

            std::vector<std::string> test_mfccs_path = {"../dataset_mfccs_raw/yes/d21fd169_nohash_0","../dataset_mfccs_raw/yes/d21fd169_nohash_0","../dataset_mfccs_raw/yes/d21fd169_nohash_1"};
            
            std::vector<float> input_mfccs = load_batch_from_paths(test_mfccs_path);
            
            std::vector<std::string> words = {"yes","no","up","down","left","right","on","off","stop","go"};
            
            std::vector<layer> model = get_model(exported_params, BATCH_SIZE, NUMBER_OF_CLASSES);

            layer softmax(LayerTypes::softmax, model.back().get_output_size());
            layer crossentropy(LayerTypes::cross_entropy_loss, softmax.get_output_size());
    //#if DO_LAYER_ANALYSIS
    //        dataloader dataloader(words, "c50f55b8_nohash_19", BATCH_SIZE, 1); // Change '/' to the userID
    //#else
    //#endif
    dataloader dataloader(words, uid, BATCH_SIZE, TRAIN_VAL_SPLIT); // Change '/' to the userID
            float error = 0.0f;
            float momentum = 0.0;
            uint32_t i = 0;
            uint32_t ii = 0;
            
            std::vector<float> all_validation_error;
            std::vector<float> all_validation_accuracies_max;
            std::vector<float> all_validation_accuracies_threshold_60;
            std::vector<float> all_validation_accuracies_threshold_65;
            std::vector<float> all_validation_accuracies_threshold_70;
            std::vector<float> all_validation_accuracies_threshold_75;
            std::vector<float> all_validation_accuracies_threshold_80;
            std::vector<float> all_validation_accuracies_threshold_85;
            std::vector<float> all_validation_accuracies_threshold_90;
            std::vector<float> all_validation_accuracies_threshold_95;
            std::vector<std::vector<float>> all_avg_train_activations;
            std::vector<std::vector<float>> all_avg_validation_activations;
            
            while (i < EPOCHS) {
                
                if (dataloader.get_training_pool_empty()) {
                    auto mybatch = dataloader.get_batch();
                    std::vector<float> labels_onehot;
                    // Convert int labels to vector of dim [batchsize classes]
                    for (auto label : std::get<1>(mybatch)) {
                        std::vector<float> temp;
                        temp = int_to_float_onehot(2 + label,12);

                        labels_onehot.insert(labels_onehot.end(), temp.begin(), temp.end());
                    }
                    if (i == 0) {
                        model_forward(model, std::get<0>(mybatch));
                        all_avg_train_activations.push_back(model[27].layer_outputs);
                    } else {
                        model[28].forward(std::get<0>(mybatch).data());
                        model[29].forward(model[28].layer_outputs.data());
                    }
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

  //  #if DO_LAYER_ANALYSIS
  //                  batchnorm_analyzer.print_stats_colnames();
  //                  batchnorm_analyzer.print_stats_raw();
  //                  conv_analyzer.print_stats_raw();
  //                  crossloss_analyzer.print_stats_raw();
  //                  dense_fw_analyzer.print_stats_raw();
  //                  dense_bw_analyzer.print_stats_raw();
  //                  dw_analyzer.print_stats_raw();
  //                  fusion_fw_analyzer.print_stats_raw();
  //                  fusion_bw_analyzer.print_stats_raw();
  //                  relu_analyzer.print_stats_raw();
  //                  softmax_fw_analyzer.print_stats_raw();
  //                  softmax_bw_analyzer.print_stats_raw();
  //  #else
  //                  dataloader.print_progress_bar(i+1,error);
//
  //  #endif

                } else {
                    momentum = 0.0;

                    auto myvalset = dataloader.get_validation_set();
                    float temp_err = 0.0f;
                    float total = 0;
                    float correct = 0;
                    float correct_validation_accuracies_threshold_60 = 0;
                    float correct_validation_accuracies_threshold_65 = 0;
                    float correct_validation_accuracies_threshold_70 = 0;
                    float correct_validation_accuracies_threshold_75 = 0;
                    float correct_validation_accuracies_threshold_80 = 0;
                    float correct_validation_accuracies_threshold_85 = 0;
                    float correct_validation_accuracies_threshold_90 = 0;
                    float correct_validation_accuracies_threshold_95 = 0;
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
                        if (i == 0) {
                            model_forward(model, actual_inputs);
                            all_avg_validation_activations.push_back(model[27].layer_outputs);
                        } else {
                            model[28].forward(actual_inputs.data());
                            model[29].forward(model[28].layer_outputs.data());
                        }
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
                            if (softmax.layer_outputs[2 + vlabel] >= 0.60f) {
                                correct_validation_accuracies_threshold_60 += 1.0f;
                            }
                            if (softmax.layer_outputs[2 + vlabel] >= 0.65f) {
                                correct_validation_accuracies_threshold_65 += 1.0f;
                            }
                            if (softmax.layer_outputs[2 + vlabel] >= 0.70f) {
                                correct_validation_accuracies_threshold_70 += 1.0f;
                            }
                            if (softmax.layer_outputs[2 + vlabel] >= 0.75f) {
                                correct_validation_accuracies_threshold_75 += 1.0f;
                            }
                            if (softmax.layer_outputs[2 + vlabel] >= 0.80f) {
                                correct_validation_accuracies_threshold_80 += 1.0f;
                            }
                            if (softmax.layer_outputs[2 + vlabel] >= 0.85f) {
                                correct_validation_accuracies_threshold_85 += 1.0f;
                            }
                            if (softmax.layer_outputs[2 + vlabel] >= 0.90f) {
                                correct_validation_accuracies_threshold_90 += 1.0f;
                            }
                            if (softmax.layer_outputs[2 + vlabel] >= 0.95f) {
                                correct_validation_accuracies_threshold_95 += 1.0f;
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
                    all_validation_error.push_back(error);
                    all_validation_accuracies_max.push_back(correct/total);
                    all_validation_accuracies_threshold_60.push_back(correct_validation_accuracies_threshold_60/total);
                    all_validation_accuracies_threshold_65.push_back(correct_validation_accuracies_threshold_65/total);
                    all_validation_accuracies_threshold_70.push_back(correct_validation_accuracies_threshold_70/total);
                    all_validation_accuracies_threshold_75.push_back(correct_validation_accuracies_threshold_75/total);
                    all_validation_accuracies_threshold_80.push_back(correct_validation_accuracies_threshold_80/total);
                    all_validation_accuracies_threshold_85.push_back(correct_validation_accuracies_threshold_85/total);
                    all_validation_accuracies_threshold_90.push_back(correct_validation_accuracies_threshold_90/total);
                    all_validation_accuracies_threshold_95.push_back(correct_validation_accuracies_threshold_95/total);
                    dataloader.reset_training_pool();
                    //std::cout << std::endl; // New line between epochs
                    if (i == 0) {
                        dataloader.set_train_set(all_avg_train_activations);
                        dataloader.set_validation_set(all_avg_validation_activations);
                    }
                    
                    i++;
                }
            }



            // Open CSV file
            std::ofstream file(validation_errors_file_name);
            
            
            // Write CSV header
            file << "Epoch,Sample_Count,Val_Loss,Val_Acc_Max,Val_Acc_Thr_60,Val_Acc_Thr_65,Val_Acc_Thr_70,Val_Acc_Thr_75,Val_Acc_Thr_80,Val_Acc_Thr_85,Val_Acc_Thr_90,Val_Acc_Thr_95" << std::endl;

            // Write error messages and codes
            for (size_t i = 0; i < all_validation_error.size(); ++i) {
                file << i <<
                "," << dataloader.get_validation_size() <<
                "," << all_validation_error[i] <<
                "," << all_validation_accuracies_max[i] <<
                "," << all_validation_accuracies_threshold_60[i] <<
                "," << all_validation_accuracies_threshold_65[i] <<
                "," << all_validation_accuracies_threshold_70[i] <<
                "," << all_validation_accuracies_threshold_75[i] <<
                "," << all_validation_accuracies_threshold_80[i] <<
                "," << all_validation_accuracies_threshold_85[i] <<
                "," << all_validation_accuracies_threshold_90[i] <<
                "," << all_validation_accuracies_threshold_95[i] <<
                std::endl;
            }

            // Close file
            file.close();
        }
    }

    return 0;
}
