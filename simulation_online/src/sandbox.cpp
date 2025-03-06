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
volatile double DENSE_BW_OUTPUT_SCALE = 5.67463019e-8;


double calculate_mse(const std::vector<float>& quant_outputs, const std::vector<float>& float_outputs) {
    if (quant_outputs.size() != float_outputs.size()) {
        std::cerr << "Error: Output size mismatch!" << std::endl;
        return -1.0;
    }

    double mse = 0.0;
    for (size_t i = 0; i < quant_outputs.size(); i++) {
        double diff = quant_outputs[i] - float_outputs[i];
        mse += diff * diff;
    }

    return mse / quant_outputs.size();
}

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
    std::vector<std::string> all_user_ids = {"4c6167ca", "c50f55b8", "3a3ee7ed", "b66f4f93", "dae01802", "ad63d93c", "63f7a489", "bd76a7fd", "f8ba7c0e", "a6d586b7", "94d370bf", "db24628d", "c22ebf46", "cd7f8c1b", "b2ae3928", "131e738d", "81332c92", "bcf614a2", "9b8a7439", "773e26f7", "472b8045", "a9ca1818", "845f8553", "6ace4fe1", "a16013b7", "9e92ef0c", "b4aa9fef", "6c968bd9", "30060aba", "cdee383b", "0d90d8e1", "c1d39ce8", "bfd26d6b", "7e7ca854", "f736ab63", "bdee441c", "39833acb", "1dc86f91", "1496195a", "b5552931", "92a9c5e6", "953fe1ad", "5c39594f", "c9b5ff26", "c7dc7278", "f5733968", "f2f0d244", "61abbf52", "195c120a", "2aca1e72", "cc6ee39b", "15c563d7", "f035e2ea", "3bfd30e6", "e3e0f145", "0137b3f4", "87d5e978", "51f7a034", "c79159aa", "1acc97de", "10ace7eb", "106a6183", "3d53244b", "e6327279", "9a76f8c3", "3c4aa5ef", "c634a189", "9151f184", "0ff728b5", "2356b88d", "a1cff772", "f2e9b610", "8281a2a8", "eeaf97c3", "beb458a4", "763188c4", "22296dbe", "d21fd169", "893705bb", "201e28a9", "b69fe0e2", "30276d03", "686d030b", "c22d3f18", "645ed69d", "ed032775", "324210dd", "85851131", "cfbedff9", "eaa83485", "e1469561", "aff582a1", "0f7266cf", "d962e5ac", "439c84f4", "c39703ec", "1a994c9f", "2e75d37a", "c4e00ee9", "8134f43f", "eb3f7d82", "332d33b1", "0b7ee1a0", "38d78313", "f19c1390", "9be15e93", "a77fbcfd", "c1e0e8e3", "8e523821", "ee1d8d49", "da040491", "7fb8d703", "95fd5a0f", "017c4098", "87070229", "5e3dde6b", "067f61e2", "51c5601d", "211b928a", "a2cc5444", "ccca5655", "e11fbc6e", "4cee0c60", "0717b9f6", "a1e71565", "da584bc0", "8769c34c", "b308773d", "ec989d6d", "0d82fd99", "bb6d4301", "8b25410a", "71d0ded4", "7211390b", "0132a06d", "6727b579", "b83c1acf", "179a61b7", "3f45a0cf", "050170cb", "888a0c49", "07089da9", "3bc21161", "b97c9f77", "a5d1becc", "4a1e736b", "e9bc5cc2", "c0445658", "bde0f20a", "0e6e36c9", "2fee065a", "bb05582b", "f798ac78", "3e31dffe", "6565a81d", "8eb4a1bf", "35d1b6ee", "f01a0d63", "d9aa8c90", "5f5eef60", "e0c782d5", "2f0ce4d9", "89947bd7", "8fe67225", "e49428d9", "d1dd03ed", "964e8cfd", "fb8c31a9", "f575faf3", "e0315cf6", "de89e2ca", "5ebc1cda", "574ec766", "98ea0818", "acfd5b5e", "e41a903b", "b29f8b23", "25e95412", "28ed6bc9", "d070ea86", "0ba018fc", "0eb48e10", "21832144", "fa57ab3b", "29fb33da", "322d17d3", "8dc18a75", "72e382bd", "20fd451f", "676f8138", "15dd287d", "56eb74ae", "679463fc", "a045368c", "525eaa62", "978240e1", "fc3ba625", "b528edb3", "57152045", "9aa21fa9", "74241b28", "3a929277", "c120e80e", "9ff2d2f4", "3a69f765", "3d794813", "b8874962", "b4ea0d9a", "cb62dbf1", "96a48d28", "b69002d4", "97f4c236", "23da904f", "37dca74f", "333d7ddb", "3589bc72", "08ab231c", "779de043", "b0ae6326", "73f20b00", "5cf1ecce", "61ab8fbc", "2927c601", "4e6902d0", "499be02e", "235b444f", "ceef6d96", "c948d727", "b76f6088", "9886d8bf", "42f81601", "171edea9", "f9643d42", "3b3d2f59", "6aa8def4", "4c3cddb8", "a04817c2", "2b5e346d", "f0659908", "d0faf7e4", "616420be", "4c6944d6", "d278d8ef", "31f01a8d", "122c5aa7", "02ade946", "890cc926", "4fd1443e", "cc4f9250", "f30285c8", "3b195250", "c6ee87a7", "742d6431", "f5626af6", "bdd22e4d", "24befdb3", "cce7416f", "a7acbbeb", "3102f006", "18e910f4", "3c257192", "ff2b842e", "0ea9c8ce", "9b02d503", "471a0925", "197f4153", "234d6a48", "61e2f74f", "d394ef8e", "7ea032f3", "8e05039f", "e7ea8b76", "3291330e", "e71a9381", "e53139ad", "f0522ff4", "2bdbe5f7", "ca4d5368", "b9f46737", "2fcb6397", "54b6d355", "f568162b", "0f46028a", "1ed0b13d", "5f9cd2eb", "321aba74", "32ad5b65", "c9b653a0", "f68160c0", "42a99aec", "d5b963aa", "1b4c9b89", "3b4f8f24", "a7200079", "f822b9bf", "c93d5e22", "e882abb2", "ef3367d9", "cd671b5f", "6ef407da", "bab36420", "229978fd", "a05a90c1", "51055bda", "b49caed3", "fa52ddf6", "70a00e98", "ac9dee0e", "617de221", "6bf5baf3", "c8b904e3", "69a1a79f", "eb0676ec", "ef77b778", "5a5721f8", "0d6d7360", "d3831f6a", "7846fd85", "f2e59fea", "a929f9b9", "099d52ad", "d264f7b6", "578d3efb", "bbc30633", "64df20d8", "29229c21", "b5cf6ea8", "ddedba85", "7213ed54", "e4be0cf6", "5170b77f", "72242187", "2da58b32", "94de6a6a", "54aecbd5", "b0f5b16d", "3cc595de", "333784b7", "1e9e6bdd", "ace072ba", "113b3fbc", "692a88e6", "14c7b073", "bbd0bbd0", "834f03fe", "833d9f56", "f5341341", "fb727898", "d98dd124", "b9515bf3", "28e47b1a", "5b09db89", "25132942", "365908bd", "1e02ffc5", "bbf38549", "cd85758f", "ce0cb033", "b87bdb22", "fbb56351", "fb7eb481", "6c429c7b", "1ecfb537", "ca4912b6", "cb72dfb6", "fbf3dd31", "06f6c194", "ffd2ba2f", "617aeb6c", "3bb68054"};
    #if DO_LAYER_ANALYSIS
#else
#pragma omp parallel for num_threads(10)
#endif
    for (auto uid : all_user_ids) {
        for (uint32_t run = 0; run < TOTAL_RUNS; run++) {
            auto validation_errors_file_name = "./user_perf_logs/"+uid+"_run_"+ std::to_string(run) +".csv";
            std::string model_params_dir = "./exported_models_fixed/";
            std::vector<std::string> test_mfccs_path = {"../dataset_mfccs_raw/yes/d21fd169_nohash_0","../dataset_mfccs_raw/yes/d21fd169_nohash_0","../dataset_mfccs_raw/yes/d21fd169_nohash_1"};
            std::vector<float> input_mfccs = load_batch_from_paths(test_mfccs_path);
            std::vector<std::string> words = {"yes","no","up","down","left","right","on","off","stop","go"};
            
            std::vector<layer_q> model = get_model_fixed(model_params_dir, BATCH_SIZE, NUMBER_OF_CLASSES);
        //    std::vector<layer> model_float = get_model_fixed_dequant_ref(model_params_dir, BATCH_SIZE, NUMBER_OF_CLASSES);
            
            quant_param_t qparam_softmax;
            qparam_softmax.scale_in             = model.back().get_qparams().scale_out;
            qparam_softmax.scale_out            = 1.0f;
            qparam_softmax.scale_weight         = 1.0f;
            qparam_softmax.weight_bits          = LAYER_SOFTMAX_QPARAM_WEIGHT_BITS;
            qparam_softmax.activation_bits      = LAYER_SOFTMAX_QPARAM_ACTIVA_BITS;
            qparam_softmax.gradient_bits        = LAYER_SOFTMAX_QPARAM_GRADNT_BITS;

            layer_q softmax(LayerTypes::softmax, model.back().get_output_size(), qparam_softmax);
        //    layer softmax_float(LayerTypes::softmax, model_float.back().get_output_size());
            layer crossentropy(LayerTypes::cross_entropy_loss, softmax.get_output_size());
#if DO_LAYER_ANALYSIS
            dataloader dataloader(words, "c50f55b8_nohash_5", BATCH_SIZE, 1); // Change "/" to the userID
#else
            //dataloader dataloader(words, uid, BATCH_SIZE, TRAIN_VAL_SPLIT); // Change "/" to the userID
            dataloader dataloader({words}, uid, BATCH_SIZE, TRAIN_VAL_SPLIT); // Change "/" to the userID

#endif
            float error = 0.0f;
            float momentum = 0.0;
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
            std::vector<std::vector<int32_t>> all_avg_train_activations;
            std::vector<std::vector<int32_t>> all_avg_validation_activations;
            
            while (i < EPOCHS) {
                
                if (dataloader.get_training_pool_empty()) {
                    auto mybatch = dataloader.get_batch_fixed();
                    std::vector<int32_t> labels_onehot;
                    std::vector<float> labels_onehot_float;
                    std::vector<int32_t> inputs;
                    std::vector<float> inputs_float;
                    for (auto label : std::get<1>(mybatch)) {
                        std::vector<int32_t> temp;
                        std::vector<float> temp_float;
                        temp = int_to_fixed_onehot(2 + label, 12);
                        temp_float = int_to_float_onehot(2 + label, 12);
                     //   std::cout << "ojfoaeijf                            " << std::to_string(label) << std::endl;
                        labels_onehot.insert(labels_onehot.end(), temp.begin(), temp.end());
                        labels_onehot_float.insert(labels_onehot_float.end(), temp_float.begin(), temp_float.end());
                    }
                    inputs = std::get<0>(mybatch);
                    for (auto input_fixed : inputs) {
                        inputs_float.push_back(input_fixed);
                    }
                    if (i == 0) {
                        model_forward(model, inputs);
                       // model_forward(model_float, inputs_float);
                        all_avg_train_activations.push_back(model[18].layer_outputs);
                    } else {
                        model[19].forward(std::get<0>(mybatch).data());
                        model[20].forward(model[19].layer_outputs.data());
                    }
                    softmax.forward(model.back().layer_outputs.data());
                //    softmax_float.forward(model_float.back().layer_outputs.data());


                    std::vector<float> softmax_outputs;

                    for (auto q :  softmax.layer_outputs) {
                        //std::cout << std::to_string(q) << std::endl;
                        softmax_outputs.push_back(static_cast<float>(q));
                    }
                    for (auto o :  labels_onehot) {
                        //std::cout << std::to_string(q) << std::endl;
                        labels_onehot_float.push_back(static_cast<float>(o));
                    }
                    crossentropy.forward(softmax_outputs.data(), labels_onehot_float.data());
                    softmax.backward(labels_onehot.data());
                    model[20].backward(softmax.layer_gradient_outputs.data()); //dense
                    model[19].backward(model[20].layer_gradient_outputs.data()); //fusion
                    //softmax_float.backward(labels_onehot_float.data());
                    //model_float[20].backward(softmax_float.layer_gradient_outputs.data()); //dense
                    //model_float[19].backward(model_float[20].layer_gradient_outputs.data()); //fusion


                    
                    /*
                    for (int iii = 0; iii < model[LAYERR].get_output_size().full; iii++) {
                        quant_outputs[iii] = static_cast<float>(model[LAYERR].layer_outputs[iii])*scale;
                        float_outputs[iii] = model_float[LAYERR].layer_outputs[iii];
                        mag_dif_outputs[iii] = model_float[LAYERR].layer_outputs[iii]/static_cast<float>(model[LAYERR].layer_outputs[iii]);
                        
                        std::cout << "Quantized: " << quant_outputs[iii] << ", Float: " << float_outputs[iii] << ", mag_dif: " << mag_dif_outputs[iii] << std::endl;
                    }
                    
                    */
                    /*
                    uint8_t LAYER = 8;
                    uint8_t LAYERR = 19;
                    auto qpram = model[LAYERR].get_qparams();
                    float scale = qpram.scale_weight;
                    double mse = 0.0;
                    int output_size = model[LAYERR].get_input_size().full;
                    
                    std::vector<float> quant_outputs(output_size);
                    std::vector<float> float_outputs(output_size);
                    std::vector<float> mag_dif_outputs(output_size);
                    
                    
                    for (int iii = 0; iii < output_size; iii++) {

                        quant_outputs[iii] = static_cast<float>(model[LAYERR].debug_float[iii])*scale;
                        float_outputs[iii] = model_float[LAYERR].debug_float[iii];
                        mag_dif_outputs[iii] = model_float[LAYERR].debug_float[iii]/(static_cast<float>(model[LAYERR].debug_float[iii])+.00001f);
                        
                        std::cout << "Quantized: " << quant_outputs[iii] << ", Float: " << float_outputs[iii] << ", mag_dif: " << mag_dif_outputs[iii] << std::endl;
                    }
                
                    mse = calculate_mse(quant_outputs, float_outputs);
                    std::cout << "MSE: " << mse << std::endl;
                    for (auto q : softmax_float.layer_outputs) {
                        std::cout << std::to_string(q) << std::endl;
                        //softmax_outputs.push_back(static_cast<float>(q));
                    }     
                    
                    for (auto q : softmax.layer_outputs) {
                        std::cout << std::to_string(q) << std::endl;
                        //softmax_outputs.push_back(static_cast<float>(q));
                    }     
                    */
                    
                        /*
                    //softmax_outputs.push_back(static_cast<float>(q));
                */
                
                   // assert(1==0);
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
              //      dataloader.print_progress_bar(i+1,error);

#endif

                } else {
                    momentum = 0.0;

                    auto myvalset = dataloader.get_validation_set_fixed();
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
                        std::vector<int32_t> labels_onehot;
                        std::vector<int32_t> actual_inputs;
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
                            all_avg_validation_activations.push_back(model[18].layer_outputs);
                        } else {
                            model[19].forward(actual_inputs.data());
                            model[20].forward(model[19].layer_outputs.data());
                        }
                        softmax.forward(model.back().layer_outputs.data());
    
    
                        std::vector<float> softmax_outputs;
                        std::vector<float> labels_onehot_float;
    
                        for (auto q :   softmax.layer_outputs) {
                            // std::cout << std::to_string(q) << std::endl;
                            softmax_outputs.push_back(static_cast<float>(q)/256);
                        }
                        for (auto o :  labels_onehot) {
                            // std::cout << "std::to_string(o)" << std::endl;
                            labels_onehot_float.push_back(static_cast<float>(o));
                        }
                        crossentropy.forward(softmax_outputs.data(), labels_onehot_float.data());
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
