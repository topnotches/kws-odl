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
    std::vector<std::string> all_user_ids = {"4c6167ca", "c50f55b8","c948d727", "b5cf6ea8", "cce7416f", "a04817c2", "b4ea0d9a", "42a99aec", "28e47b1a", "472b8045", "1b4c9b89", "cd671b5f", "de89e2ca", "953fe1ad", "ef3367d9", "893705bb", "c22d3f18", "8134f43f", "e1469561", "b66f4f93", "324210dd", "24befdb3", "229978fd", "9886d8bf", "179a61b7", "5ebc1cda", "617de221", "21832144", "2aca1e72", "8e05039f", "cd7f8c1b", "c1e0e8e3", "ddedba85", "eb0676ec", "9aa21fa9", "c1d39ce8", "5f9cd2eb", "56eb74ae"};
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
