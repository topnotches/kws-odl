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

volatile double DENSE_BW_OUTPUT_SCALE = 3.67463019e-8;


double calculate_mse(const std::vector<float>& quant_outputs, const std::vector<float>& float_outputs) {
    if (quant_outputs.size() != float_outputs.size()) {
        std::cerr << "Error: Output size mismatch!" << std::endl;
        return -1.0;
    }

    double mse = 0.0;
    for (size_t i = 0; i < quant_outputs.size(); i++) {
        double diff = static_cast<double>(quant_outputs[i]) - static_cast<double>(float_outputs[i]);
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
    std::vector<std::string> all_user_ids = {"69a1a79f", "21832144", "525eaa62", "b528edb3", "a16013b7", "42f81601", "a7200079", "b5cf6ea8", "b66f4f93", "ac9dee0e", "2aca1e72", "87d5e978", "c22d3f18", "5cf1ecce", "ca4d5368", "099d52ad", "3e31dffe", "cb72dfb6", "7846fd85", "9aa21fa9", "25e95412", "ad63d93c", "3bfd30e6", "5b09db89", "e1469561", "eaa83485", "0717b9f6", "fa52ddf6", "890cc926", "cd7f8c1b", "c39703ec", "72e382bd", "692a88e6", "b69002d4", "834f03fe", "8eb4a1bf", "e41a903b", "122c5aa7", "cb62dbf1", "773e26f7", "cce7416f", "a9ca1818", "f5341341", "fc3ba625", "ed032775", "bdee441c", "ca4912b6", "a77fbcfd", "c120e80e", "888a0c49", "8e05039f", "71d0ded4", "ddedba85", "b5552931", "686d030b", "37dca74f", "eb0676ec", "61abbf52", "3b195250", "10ace7eb", "5ebc1cda", "e11fbc6e", "d070ea86", "98ea0818", "471a0925", "e53139ad", "8b25410a", "9886d8bf", "d3831f6a", "f575faf3", "cd85758f", "fbf3dd31", "d9aa8c90", "4a1e736b", "85851131", "229978fd", "c1d39ce8", "72242187", "b4ea0d9a", "02ade946", "5170b77f", "195c120a", "a045368c", "24befdb3", "bbd0bbd0", "1dc86f91", "ccca5655", "333784b7", "56eb74ae", "51c5601d", "dae01802", "f68160c0", "b0ae6326", "179a61b7", "e4be0cf6", "28ed6bc9", "57152045", "676f8138", "c1e0e8e3", "2b5e346d", "beb458a4", "06f6c194", "0ea9c8ce", "0132a06d", "8134f43f", "0137b3f4", "c0445658", "067f61e2", "a05a90c1", "953fe1ad", "cd671b5f", "de89e2ca", "321aba74", "893705bb", "1496195a", "472b8045", "c4e00ee9", "42a99aec", "c9b653a0", "74241b28", "322d17d3", "7213ed54", "d278d8ef", "61ab8fbc", "a929f9b9", "c948d727", "499be02e", "3bc21161", "2fee065a", "7fb8d703", "1b4c9b89", "0ff728b5", "4cee0c60", "5f9cd2eb", "fbb56351", "f9643d42", "8dc18a75", "f822b9bf", "fb8c31a9", "eeaf97c3", "6727b579", "70a00e98", "324210dd", "bde0f20a", "c79159aa", "94d370bf", "c7dc7278", "ace072ba", "64df20d8", "31f01a8d", "97f4c236", "9e92ef0c", "b76f6088", "ef3367d9", "9ff2d2f4", "2da58b32", "b83c1acf", "6565a81d", "f5626af6", "964e8cfd", "61e2f74f", "a04817c2", "92a9c5e6", "617de221", "4e6902d0", "3c257192", "2927c601", "28e47b1a", "ec989d6d", "e9bc5cc2", "18e910f4", "b97c9f77", "4c3cddb8", "617aeb6c"};
    std::vector<double> all_lowest_mse_scales;

#pragma omp parallel for num_threads(1)
    for (auto uid : all_user_ids) {
        double sweep_dense_bw_scale = 0.0001f;
        auto validation_errors_file_name = "./user_finetune_dense_scale_logs/"+uid+"_dense_scales" + ".csv";
        std::string model_params_dir = "./fixed_export_run_qat_pw_w4a8_rest_w8a8_b128s4/";
        std::vector<std::string> test_mfccs_path = {"../dataset_mfccs_raw/yes/d21fd169_nohash_0","../dataset_mfccs_raw/yes/d21fd169_nohash_0","../dataset_mfccs_raw/yes/d21fd169_nohash_1"};
        std::vector<float> input_mfccs = load_batch_from_paths(test_mfccs_path);
        std::vector<std::string> words = {"yes","no","up","down","left","right","on","off","stop","go"};
        double lowest_mse = 100.0f;
        double lowest_mse_scale;
        std::vector<layer_q> model = get_model_fixed(model_params_dir, BATCH_SIZE, NUMBER_OF_CLASSES);
        std::vector<layer> model_float = get_model_fixed_dequant_ref(model_params_dir, BATCH_SIZE, NUMBER_OF_CLASSES);
        
        quant_param_t qparam_softmax;
        qparam_softmax.scale_in             = model.back().get_qparams().scale_out;
        qparam_softmax.scale_out            = 1.0f;
        qparam_softmax.scale_weight         = 1.0f;
        qparam_softmax.weight_bits          = LAYER_SOFTMAX_QPARAM_WEIGHT_BITS;
        qparam_softmax.activation_bits      = LAYER_SOFTMAX_QPARAM_ACTIVA_BITS;
        qparam_softmax.gradient_bits        = LAYER_SOFTMAX_QPARAM_GRADNT_BITS;

        layer_q softmax(LayerTypes::softmax, model.back().get_output_size(), qparam_softmax);
        layer softmax_float(LayerTypes::softmax, model_float.back().get_output_size());
        layer crossentropy(LayerTypes::cross_entropy_loss, softmax.get_output_size());

        dataloader dataloader({words}, uid, BATCH_SIZE, 1.0f, "../dataset_mfccs_raw/", false); // Change "/" to the userID

        float error = 0.0f;
        float momentum = 0.0;
        uint32_t i = 0;
        std::vector<double> all_grad_out_scale_mse_error;
        std::vector<double> all_grad_out_scale;
        std::vector<std::vector<int32_t>> all_avg_train_activations;
        std::vector<std::vector<float>>   all_avg_train_activations_float;

        std::vector<float> quant_outputs;
        std::vector<float> float_outputs;
        std::vector<float> mag_dif_outputs;
        for (auto layer: model) {
            std::cout << layer.get_rescale_value() << std::endl;
        }
        assert( 1 == 0 );
        while (i < EPOCHS) {

            if (dataloader.get_training_pool_empty()) {
                model[20].set_dense_bw_output_scale(sweep_dense_bw_scale);

                if (i == 0) {
                    auto mybatch = dataloader.get_batch_fixed();
                    std::vector<int32_t> labels_onehot;
                    std::vector<float> labels_onehot_float;
                    std::vector<int32_t> inputs;
                    std::vector<float> inputs_float;
                    
                    for (auto label : std::get<1>(mybatch)) {
                        std::vector<int32_t> temp;
                        std::vector<float> temp_float;

                       // std::cout << std::to_string(label) << std::endl;
                        temp = int_to_fixed_onehot(2 + label, 12);
                        temp_float = int_to_float_onehot(2 + label, 12);
                        labels_onehot.insert(labels_onehot.end(), temp.begin(), temp.end());
                        labels_onehot_float.insert(labels_onehot_float.end(), temp_float.begin(), temp_float.end());
                    }
                    inputs = std::get<0>(mybatch);
                    for (auto input_fixed : inputs) {
                        inputs_float.push_back(input_fixed);
                    }
    
                    model_forward(model, inputs);
                    model_forward(model_float, inputs_float);

                    all_avg_train_activations.insert(all_avg_train_activations.begin(), model[18].layer_outputs);
                    all_avg_train_activations_float.insert(all_avg_train_activations_float.begin(), model_float[18].layer_outputs);

                    if (all_avg_train_activations.size() != all_avg_train_activations_float.size()) {
                        assert (1==9);
                    }
                    softmax.forward(model.back().layer_outputs.data());
                    softmax_float.forward(model_float.back().layer_outputs.data());
    
                    for (auto o :  labels_onehot) {
                        labels_onehot_float.push_back(static_cast<float>(o));
                    }
    
    
                    softmax.backward(labels_onehot.data());
                    model[20].backward(softmax.layer_gradient_outputs.data()); //dense
    
                    softmax_float.backward(labels_onehot_float.data());
                    model_float[20].backward(softmax_float.layer_gradient_outputs.data()); //dense

                } else {
                    std::tuple<std::tuple<std::vector<int32_t>,std::vector<uint8_t>>, std::tuple<std::vector<float>,std::vector<uint8_t>>> mybatch_mix = dataloader.get_batch_fixed_and_float();
                    std::tuple<std::vector<int32_t>,std::vector<uint8_t>> mybatch_fixed = std::get<0>(mybatch_mix);
                    std::tuple<std::vector<float>,std::vector<uint8_t>> mybatch_float = std::get<1>(mybatch_mix);
                    std::vector<int32_t> labels_onehot;
                    std::vector<float> labels_onehot_float;
                    std::vector<int32_t> inputs;
                    std::vector<float> inputs_float;
                    for (auto label : std::get<1>(mybatch_fixed)) {
                       // std::cout << std::to_string(label) << std::endl;

                        std::vector<int32_t> temp;
                        std::vector<float> temp_float;
                        temp = int_to_fixed_onehot(2 + label, 12);
                        temp_float = int_to_float_onehot(2 + label, 12);
                        labels_onehot.insert(labels_onehot.end(), temp.begin(), temp.end());
                        labels_onehot_float.insert(labels_onehot_float.end(), temp_float.begin(), temp_float.end());
                    }
    
                    model[19].forward(std::get<0>(mybatch_fixed).data());
                    model[20].forward(model[19].layer_outputs.data());
                    softmax.forward(model.back().layer_outputs.data());

                    model_float[19].forward(std::get<0>(mybatch_float).data());
                    model_float[20].forward(model_float[19].layer_outputs.data());
                    softmax_float.forward(model_float.back().layer_outputs.data());
    
                    for (auto o :  labels_onehot) {
                        //std::cout << std::to_string(q) << std::endl;
                        labels_onehot_float.push_back(static_cast<float>(o));
                    }
    
    
                    softmax.backward(labels_onehot.data());
                    model[20].backward(softmax.layer_gradient_outputs.data()); //dense
    
                    softmax_float.backward(labels_onehot_float.data());
                    model_float[20].backward(softmax_float.layer_gradient_outputs.data()); //dense
                }



                uint8_t LAYER = 8;
                uint8_t LAYERR = 20;
                auto qpram = model[LAYERR].get_qparams();
                double scale = sweep_dense_bw_scale;
                double mse = 0.0;
                int output_size = model[LAYERR].get_input_size().full;
                
                
                
                for (int iii = 0; iii < output_size; iii++) {

                    quant_outputs.push_back(static_cast<float>(model[LAYERR].layer_gradient_outputs[iii])*sweep_dense_bw_scale);
                    float_outputs.push_back(model_float[LAYERR].layer_gradient_outputs[iii]);
                    mag_dif_outputs.push_back(model_float[LAYERR].layer_gradient_outputs[iii]/(static_cast<float>(model[LAYERR].layer_gradient_outputs[iii])+.00001f));
                    
                   // std::cout << model[19].layer_outputs[iii] << " " << model_float[19].layer_outputs[iii] << ", Scale "<< DENSE_BW_OUTPUT_SCALE << ", Quantized: " << quant_outputs[iii] << ", Float: " << float_outputs[iii] << ", mag_dif: " << mag_dif_outputs[iii] << std::endl;
                }
            

            } else {

                momentum = 0.0;
                double mse = calculate_mse(quant_outputs, float_outputs);
                if (mse < lowest_mse) {
                    lowest_mse = mse;
                    lowest_mse_scale = sweep_dense_bw_scale;
                }
                all_grad_out_scale.push_back(static_cast<double>(sweep_dense_bw_scale));
                all_grad_out_scale_mse_error.push_back(mse);
                dataloader.reset_training_pool();
               // std::cout << std::endl; // New line between epochs
                if (i == 0) {
                    dataloader.set_train_set(all_avg_train_activations);
                    dataloader.set_train_set(all_avg_train_activations_float);
                }
                sweep_dense_bw_scale *= 0.99;
                quant_outputs.clear();
                float_outputs.clear();
                mag_dif_outputs.clear();
                i++;
            }
        }


        
        // Open CSV file
        std::ofstream file(validation_errors_file_name);
        
        
        // Write CSV header
        file << "SCALE,Sample_Count,MSE" << std::endl;
        
        // Write error messages and codes
        for (size_t i = 0; i < all_grad_out_scale_mse_error.size(); ++i) {
            file << all_grad_out_scale[i] <<
            "," << all_grad_out_scale_mse_error[i] <<
            std::endl;
        }
        
        // Close file
        file.close();
        all_lowest_mse_scales.push_back(lowest_mse_scale);

    }

    double sum_scales = 0.0f;

    for (double scale : all_lowest_mse_scales) {
        std::cout << scale << std::endl;
        sum_scales += scale;
    }
    double mean_scale = sum_scales/static_cast<double>(all_lowest_mse_scales.size());
    std::cout << "Mean output scale: " << mean_scale << std::endl;

    double numerator = 0.0f;
    for (double scale : all_lowest_mse_scales) {
        double temp = scale - mean_scale;
        numerator += scale*scale;
    }

    double var_scale = numerator/static_cast<double>(all_lowest_mse_scales.size()-1);

    std::cout << "Variance:          " << var_scale << std::endl;
    std::cout << "StDv:              " << sqrt(var_scale) << std::endl;

    return 0;
}
