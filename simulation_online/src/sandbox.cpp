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
        auto temp_floats = load_mffcs_bin_float(path);
        input_mfccs.insert(input_mfccs.end(), temp_floats.begin(), temp_floats.end());
    }
    return input_mfccs;
}

int  main() {
    std::vector<std::string> all_user_ids = {"eat"};//{"4c6167ca", "c50f55b8", "4fd1443e", "31f01a8d", "3a69f765", "4cee0c60", "b29f8b23", "2356b88d", "179a61b7", "833d9f56", "fbf3dd31", "a1cff772", "333d7ddb", "fb8c31a9", "61e2f74f", "bbd0bbd0", "bbf38549", "c79159aa", "3b3d2f59", "b4ea0d9a", "834f03fe", "0ea9c8ce", "bde0f20a", "eeaf97c3", "2da58b32", "234d6a48", "890cc926", "616420be", "a5d1becc", "229978fd", "3102f006", "9be15e93", "61ab8fbc", "3d53244b", "773e26f7", "3291330e", "1a994c9f", "d3831f6a", "b528edb3", "8fe67225", "18e910f4", "617de221", "a05a90c1", "97f4c236", "b4aa9fef", "f2e9b610", "92a9c5e6", "439c84f4", "cc4f9250", "578d3efb", "3cc595de", "f30285c8", "20fd451f", "94d370bf", "73f20b00", "b308773d", "0e6e36c9", "30276d03", "b69002d4", "29229c21", "95fd5a0f", "1ecfb537", "f0522ff4", "893705bb", "574ec766", "56eb74ae", "472b8045", "d962e5ac", "322d17d3", "3b4f8f24", "365908bd", "f568162b", "9b02d503", "4c6944d6", "cfbedff9", "692a88e6", "8b25410a", "5cf1ecce", "81332c92", "7846fd85", "cd85758f", "099d52ad", "a1e71565", "2fcb6397", "888a0c49", "324210dd", "7e7ca854", "e0315cf6", "b66f4f93", "87d5e978", "61abbf52", "a7acbbeb", "b83c1acf", "fa52ddf6", "e882abb2", "c0445658", "3589bc72", "e1469561", "201e28a9", "fb727898", "c8b904e3", "f2f0d244", "c1e0e8e3", "dae01802", "0ba018fc", "94de6a6a", "aff582a1", "29fb33da", "c1d39ce8", "30060aba", "f01a0d63", "5e3dde6b", "72242187", "ffd2ba2f", "d0faf7e4", "1e02ffc5", "1b4c9b89", "bbc30633", "779de043", "f8ba7c0e", "89947bd7", "1dc86f91", "f5733968", "de89e2ca", "d9aa8c90", "fbb56351", "72e382bd", "017c4098", "332d33b1", "bd76a7fd", "106a6183", "c6ee87a7", "7213ed54", "9aa21fa9", "28e47b1a", "4a1e736b", "25e95412", "d98dd124", "0d6d7360", "0d82fd99", "6727b579", "06f6c194", "b69fe0e2", "b49caed3", "b0ae6326", "71d0ded4", "1ed0b13d", "e9bc5cc2", "69a1a79f", "02ade946", "15c563d7", "3c257192", "bab36420", "e11fbc6e", "3a929277", "db24628d", "2e75d37a", "d264f7b6", "51055bda", "e53139ad", "ef77b778", "acfd5b5e", "2fee065a", "6c968bd9", "a2cc5444", "c39703ec", "2f0ce4d9", "9b8a7439", "8e05039f", "5b09db89", "25132942", "d070ea86", "b97c9f77", "28ed6bc9", "fa57ab3b", "0d90d8e1", "ddedba85", "d21fd169", "6ace4fe1", "bcf614a2", "ad63d93c", "21832144", "e71a9381", "197f4153", "7211390b", "8281a2a8", "d5b963aa", "471a0925", "122c5aa7", "1e9e6bdd", "42f81601", "f822b9bf", "7ea032f3", "8e523821", "f798ac78", "70a00e98", "b5cf6ea8", "bb05582b", "0ff728b5", "f575faf3", "2b5e346d", "24befdb3", "8769c34c", "cce7416f", "6bf5baf3", "b2ae3928", "a16013b7", "eaa83485", "763188c4", "525eaa62", "845f8553", "ee1d8d49", "42a99aec", "15dd287d", "645ed69d", "10ace7eb", "57152045", "b9f46737", "171edea9", "ce0cb033", "1acc97de", "eb3f7d82", "cd671b5f", "f035e2ea", "3e31dffe", "7fb8d703", "3c4aa5ef", "a77fbcfd", "ed032775", "5170b77f", "f9643d42", "a04817c2", "0f7266cf", "3bc21161", "f2e59fea", "3bb68054", "51c5601d", "ca4d5368", "37dca74f", "bfd26d6b", "c9b653a0", "9e92ef0c", "235b444f", "679463fc", "e49428d9", "c7dc7278", "c634a189", "978240e1", "32ad5b65", "23da904f", "ace072ba", "e4be0cf6", "3bfd30e6", "f5626af6", "ec989d6d", "2bdbe5f7", "953fe1ad", "4c3cddb8", "676f8138", "ccca5655", "63f7a489", "cc6ee39b", "d1dd03ed", "050170cb", "0eb48e10", "85851131", "6565a81d", "5a5721f8", "64df20d8", "0b7ee1a0", "b0f5b16d", "8dc18a75", "c9b5ff26", "07089da9", "87070229", "211b928a", "f0659908", "0132a06d", "cb62dbf1", "08ab231c", "96a48d28", "a9ca1818", "54aecbd5", "39833acb", "38d78313", "51f7a034", "9151f184", "c22d3f18", "14c7b073", "b5552931", "5f9cd2eb", "6c429c7b", "b76f6088", "c4e00ee9", "617aeb6c", "067f61e2", "9886d8bf", "742d6431", "321aba74", "c93d5e22", "c120e80e", "9a76f8c3", "bdee441c", "f68160c0", "ff2b842e", "beb458a4", "da040491", "b8874962", "b87bdb22", "ef3367d9", "cdee383b", "195c120a", "a929f9b9", "bb6d4301", "d278d8ef", "c22ebf46", "da584bc0", "e0c782d5", "6aa8def4", "22296dbe", "cd7f8c1b", "74241b28", "c948d727", "3a3ee7ed", "b9515bf3", "54b6d355", "964e8cfd", "eb0676ec", "98ea0818", "d394ef8e", "f5341341", "fb7eb481", "35d1b6ee", "131e738d", "f736ab63", "8eb4a1bf", "a045368c", "e6327279", "5ebc1cda", "a6d586b7", "3d794813", "a7200079", "5c39594f", "9ff2d2f4", "2aca1e72", "0137b3f4", "1496195a", "8134f43f", "2927c601", "ca4912b6", "4e6902d0", "5f5eef60", "e41a903b", "bdd22e4d", "ceef6d96", "ac9dee0e", "0717b9f6", "f19c1390", "fc3ba625", "0f46028a", "3b195250", "333784b7", "6ef407da", "499be02e", "3f45a0cf", "e3e0f145", "cb72dfb6", "e7ea8b76", "686d030b", "113b3fbc"};
#if DO_LAYER_ANALYSIS
#else
//#pragma omp parallel for num_threads(10)
#endif
    for (auto uid : all_user_ids) {
        for (uint32_t run = 0; run < TOTAL_RUNS; run++) {
            auto validation_errors_file_name = "./user_perf_logs/"+uid+"_run_"+ std::to_string(run) +".csv";
            std::string model_params_dir = "./exported_models_fixed/";
            std::vector<std::string> test_mfccs_path = {"../dataset_mfccs_raw/yes/d21fd169_nohash_0","../dataset_mfccs_raw/yes/d21fd169_nohash_0","../dataset_mfccs_raw/yes/d21fd169_nohash_1"};
            std::vector<float> input_mfccs = load_batch_from_paths(test_mfccs_path);
            std::vector<std::string> words = {"yes","no","up","down","left","right","on","off","stop","go"};
            
            std::vector<layer_q> model = get_model_fixed(model_params_dir, BATCH_SIZE, NUMBER_OF_CLASSES);
            
            quant_param_t qparam_softmax;
            qparam_softmax.scale_in             = model.back().get_qparams().scale_out;
            qparam_softmax.scale_out            = 1;
            qparam_softmax.scale_weight         = 1;
            qparam_softmax.weight_bits          = LAYER_SOFTMAX_QPARAM_WEIGHT_BITS;
            qparam_softmax.activation_bits      = LAYER_SOFTMAX_QPARAM_ACTIVA_BITS;

            layer_q softmax(LayerTypes::softmax, model.back().get_output_size(), qparam_softmax);
            //layer_q crossentropy(LayerTypes::cross_entropy_loss, softmax.get_output_size());
#if DO_LAYER_ANALYSIS
            dataloader dataloader(words, "c50f55b8_nohash_19", BATCH_SIZE, 1); // Change '/' to the userID
#else
            //dataloader dataloader(words, uid, BATCH_SIZE, TRAIN_VAL_SPLIT); // Change '/' to the userID
            dataloader dataloader({"no"}, "c50f55b8_nohash_3", BATCH_SIZE, 1); // Change '/' to the userID

#endif
            //float error = 0.0f;
            //float momentum = 0.0;
            uint32_t i = 0;
            
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
            
            while (i < 1) {
                
                if (dataloader.get_training_pool_empty()) {
                    auto mybatch = dataloader.get_batch_fixed();
                    std::vector<int32_t> labels_onehot;
                    for (auto label : std::get<1>(mybatch)) {
                        std::vector<int32_t> temp;
                        temp = int_to_fixed_onehot(2 + label, 12);

                        labels_onehot.insert(labels_onehot.end(), temp.begin(), temp.end());
                    }
                    if (i == 0) {
                        model_forward(model, std::get<0>(mybatch));
                        //all_avg_train_activations.push_back(model[27].layer_outputs);
                        /*
                        for (auto layer : model) {
                            layer.print_layer_type();
                            std::cout << "scale_in:        " << std::to_string(layer.get_qparams().scale_in) << std::endl;
                            std::cout << "scale_out:       " << std::to_string(layer.get_qparams().scale_out) << std::endl;
                            std::cout << "scale_weight:    " << std::to_string(layer.get_qparams().scale_weight) << std::endl;
                            std::cout << "weight_bits:     " << std::to_string(layer.get_qparams().weight_bits) << std::endl;
                            std::cout << "activation_bits: " << std::to_string(layer.get_qparams().activation_bits) << std::endl;
                            std::cout << "rescale_value:   " << std::to_string(layer.get_rescale_value()) << std::endl;
                        }
                        for (auto q :  model[11].layer_outputs) {
                            std::cout << std::to_string(q) << std::endl;
                        }
                        */
                    } else {
                        //model[28].forward(std::get<0>(mybatch).data());
                        //model[29].forward(model[28].layer_outputs.data());
                    }
                    softmax.forward(model.back().layer_outputs.data());
                    for (auto q :  softmax.layer_outputs) {
                        std::cout << std::to_string(q) << std::endl;
                    }
                    //crossentropy.forward(softmax.layer_outputs.data(), labels_onehot.data());
                    //softmax.backward(labels_onehot.data());
                    //model[29].backward(softmax.layer_gradient_outputs.data()); //dense
                    //model[28].backward(model[29].layer_gradient_outputs.data()); //fusion
                    //float temp_err = 0.0f;
//
                    //for (auto output : crossentropy.layer_outputs) {
                    //    temp_err += output;
                    //}
                    //temp_err /= crossentropy.layer_outputs.size()/BATCH_SIZE;

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
                    /*
                    momentum = 0.0;

                    auto myvalset = dataloader.get_validation_set_float();
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
                        //softmax.forward(model.back().layer_outputs.data());
                        //crossentropy.forward(softmax.layer_outputs.data(), labels_onehot.data());
                        
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
                    
                    */
                    i++;
                }
            }


            /*
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
            */

        }
    }

    return 0;
}
