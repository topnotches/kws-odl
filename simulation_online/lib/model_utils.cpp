#include "model_utils.hpp"
#include "misc_utils.hpp"
#include "defs.hpp"
#include "layer.hpp"
#include <sstream>
#include <vector>
#include <stdint.h>
#include <iostream>
#include <fstream>

std::vector<float> str_to_fl32_vec(std::string string_of_floats, std::string delimiter) {
    std::vector<float> float_vector;

    auto position = string_of_floats.find(delimiter);


    while (position != std::string::npos) {

        
        float_vector.push_back(std::stof(string_of_floats.substr(0, position)));
        string_of_floats.erase(0, position + delimiter.length());

        position = string_of_floats.find(delimiter);
    }
    return float_vector;
}

conv_hypr_param_t set_conv_param(uint8_t width, uint8_t height, uint8_t stride, uint8_t count, uint8_t tp, uint8_t bp, uint8_t lp, uint8_t rp) {
    conv_hypr_param_t params;

    params.pad_top = tp;
    params.pad_bottom = bp;
    params.pad_left = lp;
    params.pad_right = rp;

    params.kernel_stride = stride;
    params.kernel_width = width;
    params.kernel_height = height;
    params.kernel_count = count;
    
    return params;
}

dense_hypr_param_t set_dense_param(uint32_t output_size, uint32_t input_size) {
    dense_hypr_param_t params;

    params.size_in = input_size;
    params.size_out = output_size;
    
    return params;
}

std::vector<std::vector<std::string>> load_layers_from_csv_to_vec(const std::string& filePath) {
    std::ifstream file(filePath);
    std::string line;
    std::vector<std::vector<std::string>> csv_data;
    int current_row = 0;
    

    while (getline(file, line)) {
        if(current_row>0) {

            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row_data, row_data_raw;
            std::string start0_ss = "\"[";
            std::string stop0_ss = "]\"";
            std::string start1_ss = "[";
            std::string stop1_ss = "]";
            std::string name;
            std::string size = "";
            std::string params = "";
            uint8_t clean_state = 0;

            while (getline(ss, cell, ',')) {
                row_data_raw.push_back(cell);
            }

            for (auto string_cell : row_data_raw) {
                switch (clean_state)
                {
                case 0:{
                    name = string_cell;

                    clean_state++;
                    break;
                }
                case 1:{

                    
                    std::string::size_type start0_find = string_cell.find(start0_ss);
                    if (start0_find != std::string::npos)
                        string_cell.erase(start0_find, start0_ss.length());
                    std::string::size_type start1_find = string_cell.find(start1_ss);
                    if (start1_find != std::string::npos)
                        string_cell.erase(start1_find, start1_ss.length());

                    std::string::size_type stop0_find = string_cell.find(stop0_ss);
                    if (stop0_find != std::string::npos) {
                        string_cell.erase(stop0_find, stop0_ss.length());
                        clean_state++;
                    }
                    std::string::size_type stop1_find = string_cell.find(stop1_ss);
                    if (stop1_find != std::string::npos) {
                        string_cell.erase(stop1_find, stop1_ss.length());
                        clean_state++;
                    }
                    size = size + string_cell;

                    break;
                }
                case 2:{

                    std::string::size_type start_find = string_cell.find(start0_ss);
                    std::string::size_type stop_find = string_cell.find(stop0_ss);
                    
                    if (start_find != std::string::npos)
                        string_cell.erase(start_find, start0_ss.length());

                    if (stop_find != std::string::npos) {
                        string_cell.erase(stop_find, stop0_ss.length());
                        clean_state++;
                    }
                    params = params + string_cell;

                    break;
                }
                
                default: {

                    break;
                }
                }
            }

            row_data.push_back(name);
            row_data.push_back(size);
            row_data.push_back(params + " ");

            csv_data.push_back(row_data);
        }
        current_row++;
    }

    file.close();
    return csv_data;
}

std::vector<layer> get_model(std::string model_path, uint8_t batch_size, uint8_t num_classes) {
    std::vector<layer> model;


    std::vector<std::vector<std::string>> layer_params_str = load_layers_from_csv_to_vec(model_path);
    std::vector<std::vector<float>> layer_params;
    for (auto layer_name: layer_params_str) {
        layer_params.push_back(str_to_fl32_vec(layer_name[2]));
    }
    tensor_dim_sizes_t mfccs_size;
    mfccs_size.height = 49;
    mfccs_size.width = 10;
    mfccs_size.depth = 1;
    mfccs_size.batch = batch_size;
    mfccs_size.full = mfccs_size.width *
                            mfccs_size.height *
                            mfccs_size.depth *
                            mfccs_size.batch;
    /**********************************************
    ***** Set hyper_parameters for each layer *****
    **********************************************/
    // Layer 1, Convolution
    conv_hypr_param_t param_conv_layer1 = set_conv_param(4,10,2,64,5,5,1,1);
    // Layer 2, Depth-wise Separable Convolution
    conv_hypr_param_t param_dw_conv_layer2 = set_conv_param(3,3,1,64,1,1,1,1);
    conv_hypr_param_t param_pw_conv_layer2 = set_conv_param(1,1,1,64,0,0,0,0);
    // Layer 3, Depth-wise Separable Convolution
    conv_hypr_param_t param_dw_conv_layer3 = set_conv_param(3,3,1,64,1,1,1,1);
    conv_hypr_param_t param_pw_conv_layer3 = set_conv_param(1,1,1,64,0,0,0,0);
    // Layer 4, Depth-wise Separable Convolution
    conv_hypr_param_t param_dw_conv_layer4 = set_conv_param(3,3,1,64,1,1,1,1);
    conv_hypr_param_t param_pw_conv_layer4 = set_conv_param(1,1,1,64,0,0,0,0);
    // Layer 5, Depth-wise Separable Convolution
    conv_hypr_param_t param_dw_conv_layer5 = set_conv_param(3,3,1,64,1,1,1,1);
    conv_hypr_param_t param_pw_conv_layer5 = set_conv_param(1,1,1,64,0,0,0,0);
    // Layer 6, Average Pooling 
    conv_hypr_param_t param_ap2d_conv_layer6 = set_conv_param(5,25,1,64,0,0,0,0);
    // Layer 7, Dense 
    dense_hypr_param_t param_dense_layer7 = set_dense_param(num_classes);

    /****************************************
    ***** Create Instantiate Each Layer *****
    ****************************************/

    // Layer 1, Convolution
    model.push_back(layer(LayerTypes::conv,         mfccs_size,                      layer_params[0].data(),  layer_params[1].data(),   param_conv_layer1));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[2].data(),  layer_params[3].data(),   {}, {}, layer_params[4].data(), layer_params[5].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    // Layer 2, Depth-wise Separable Convolution
    model.push_back(layer(LayerTypes::dw_conv,      model.back().get_output_size(),  layer_params[6].data(),  layer_params[7].data(),   param_dw_conv_layer2));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[8].data(),  layer_params[9].data(),   {}, {}, layer_params[10].data(), layer_params[11].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    model.push_back(layer(LayerTypes::conv,         model.back().get_output_size(),  layer_params[12].data(),  layer_params[13].data(),   param_pw_conv_layer2));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[14].data(), layer_params[15].data(),  {}, {}, layer_params[16].data(), layer_params[17].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));

    // Layer 3, Depth-wise Separable Convolution
    model.push_back(layer(LayerTypes::dw_conv,      model.back().get_output_size(),  layer_params[18].data(), layer_params[19].data(),  param_dw_conv_layer3));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[20].data(),  layer_params[21].data(), {}, {}, layer_params[22].data(), layer_params[23].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    model.push_back(layer(LayerTypes::conv,         model.back().get_output_size(),  layer_params[24].data(),  layer_params[25].data(), param_pw_conv_layer3));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[26].data(),  layer_params[27].data(), {}, {}, layer_params[28].data(), layer_params[29].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));

    // Layer 4, Depth-wise Separable Convolution
    model.push_back(layer(LayerTypes::dw_conv,      model.back().get_output_size(),  layer_params[30].data(),  layer_params[31].data(), param_dw_conv_layer4));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[32].data(),  layer_params[33].data(), {}, {}, layer_params[34].data(), layer_params[35].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    model.push_back(layer(LayerTypes::conv,         model.back().get_output_size(),  layer_params[36].data(),  layer_params[37].data(), param_pw_conv_layer4));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[38].data(),  layer_params[39].data(), {}, {}, layer_params[40].data(), layer_params[41].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));

    // Layer 5, Depth-wise Separable Convolution
    model.push_back(layer(LayerTypes::dw_conv,      model.back().get_output_size(),  layer_params[42].data(),  layer_params[43].data(), param_dw_conv_layer5));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[44].data(),  layer_params[45].data(), {}, {}, layer_params[46].data(), layer_params[47].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    model.push_back(layer(LayerTypes::conv,         model.back().get_output_size(),  layer_params[48].data(),  layer_params[49].data(), param_pw_conv_layer5));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[50].data(),  layer_params[51].data(), {}, {}, layer_params[52].data(), layer_params[53].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    
    // Layer 6, Average Pooling
    model.push_back(layer(LayerTypes::avgpool2d,    model.back().get_output_size(), {}, {}, param_ap2d_conv_layer6));
    
    // Layer 7, fusion
    model.push_back(layer(LayerTypes::fusion,       model.back().get_output_size()));

    // Layer 8, Dense 
    model.push_back(layer(LayerTypes::dense,        model.back().get_output_size(), layer_params[54].data(),  layer_params[55].data(), {}, param_dense_layer7));
    return model;
}

std::vector<layer> get_model_fixed(std::string model_path, uint8_t batch_size, uint8_t num_classes) {
    std::vector<layer> model;


    std::vector<std::vector<std::string>> layer_params_str = load_layers_from_csv_to_vec(model_path);
    std::vector<std::vector<float>> layer_params;
    for (auto layer_name: layer_params_str) {
        layer_params.push_back(str_to_fl32_vec(layer_name[2]));
    }
    tensor_dim_sizes_t mfccs_size;
    mfccs_size.height = 49;
    mfccs_size.width = 10;
    mfccs_size.depth = 1;
    mfccs_size.batch = batch_size;
    mfccs_size.full = mfccs_size.width *
                            mfccs_size.height *
                            mfccs_size.depth *
                            mfccs_size.batch;
    /**********************************************
    ***** Set hyper_parameters for each layer *****
    **********************************************/
    // Layer 1, Convolution
    conv_hypr_param_t param_conv_layer1 = set_conv_param(4,10,2,64,5,5,1,1);
    // Layer 2, Depth-wise Separable Convolution
    conv_hypr_param_t param_dw_conv_layer2 = set_conv_param(3,3,1,64,1,1,1,1);
    conv_hypr_param_t param_pw_conv_layer2 = set_conv_param(1,1,1,64,0,0,0,0);
    // Layer 3, Depth-wise Separable Convolution
    conv_hypr_param_t param_dw_conv_layer3 = set_conv_param(3,3,1,64,1,1,1,1);
    conv_hypr_param_t param_pw_conv_layer3 = set_conv_param(1,1,1,64,0,0,0,0);
    // Layer 4, Depth-wise Separable Convolution
    conv_hypr_param_t param_dw_conv_layer4 = set_conv_param(3,3,1,64,1,1,1,1);
    conv_hypr_param_t param_pw_conv_layer4 = set_conv_param(1,1,1,64,0,0,0,0);
    // Layer 5, Depth-wise Separable Convolution
    conv_hypr_param_t param_dw_conv_layer5 = set_conv_param(3,3,1,64,1,1,1,1);
    conv_hypr_param_t param_pw_conv_layer5 = set_conv_param(1,1,1,64,0,0,0,0);
    // Layer 6, Average Pooling 
    conv_hypr_param_t param_ap2d_conv_layer6 = set_conv_param(5,25,1,64,0,0,0,0);
    // Layer 7, Dense 
    dense_hypr_param_t param_dense_layer7 = set_dense_param(num_classes);

    /****************************************
    ***** Create Instantiate Each Layer *****
    ****************************************/

    // Layer 1, Convolution
    model.push_back(layer(LayerTypes::conv,         mfccs_size,                      layer_params[0].data(),  layer_params[1].data(),   param_conv_layer1));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[2].data(),  layer_params[3].data(),   {}, {}, layer_params[4].data(), layer_params[5].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    // Layer 2, Depth-wise Separable Convolution
    model.push_back(layer(LayerTypes::dw_conv,      model.back().get_output_size(),  layer_params[6].data(),  layer_params[7].data(),   param_dw_conv_layer2));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[8].data(),  layer_params[9].data(),   {}, {}, layer_params[10].data(), layer_params[11].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    model.push_back(layer(LayerTypes::conv,         model.back().get_output_size(),  layer_params[12].data(),  layer_params[13].data(),   param_pw_conv_layer2));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[14].data(), layer_params[15].data(),  {}, {}, layer_params[16].data(), layer_params[17].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));

    // Layer 3, Depth-wise Separable Convolution
    model.push_back(layer(LayerTypes::dw_conv,      model.back().get_output_size(),  layer_params[18].data(), layer_params[19].data(),  param_dw_conv_layer3));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[20].data(),  layer_params[21].data(), {}, {}, layer_params[22].data(), layer_params[23].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    model.push_back(layer(LayerTypes::conv,         model.back().get_output_size(),  layer_params[24].data(),  layer_params[25].data(), param_pw_conv_layer3));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[26].data(),  layer_params[27].data(), {}, {}, layer_params[28].data(), layer_params[29].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));

    // Layer 4, Depth-wise Separable Convolution
    model.push_back(layer(LayerTypes::dw_conv,      model.back().get_output_size(),  layer_params[30].data(),  layer_params[31].data(), param_dw_conv_layer4));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[32].data(),  layer_params[33].data(), {}, {}, layer_params[34].data(), layer_params[35].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    model.push_back(layer(LayerTypes::conv,         model.back().get_output_size(),  layer_params[36].data(),  layer_params[37].data(), param_pw_conv_layer4));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[38].data(),  layer_params[39].data(), {}, {}, layer_params[40].data(), layer_params[41].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));

    // Layer 5, Depth-wise Separable Convolution
    model.push_back(layer(LayerTypes::dw_conv,      model.back().get_output_size(),  layer_params[42].data(),  layer_params[43].data(), param_dw_conv_layer5));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[44].data(),  layer_params[45].data(), {}, {}, layer_params[46].data(), layer_params[47].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    model.push_back(layer(LayerTypes::conv,         model.back().get_output_size(),  layer_params[48].data(),  layer_params[49].data(), param_pw_conv_layer5));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[50].data(),  layer_params[51].data(), {}, {}, layer_params[52].data(), layer_params[53].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    
    // Layer 6, Average Pooling
    model.push_back(layer(LayerTypes::avgpool2d,    model.back().get_output_size(), {}, {}, param_ap2d_conv_layer6));
    
    // Layer 7, fusion
    model.push_back(layer(LayerTypes::fusion,       model.back().get_output_size()));

    // Layer 8, Dense 
    model.push_back(layer(LayerTypes::dense,        model.back().get_output_size(), layer_params[54].data(),  layer_params[55].data(), {}, param_dense_layer7));
    return model;
}


void model_forward(std::vector<layer> &model, std::vector<float> data) {
    model[0].forward(data.data());
    for (uint8_t layer_index = 1; layer_index < model.size(); layer_index++ ) {
       model[layer_index].forward(model[layer_index - 1].layer_outputs.data());
       //printf("size [%d %d %d %d]\n", model[layer_index-1].get_input_size().batch, model[layer_index-1].get_input_size().depth, model[layer_index-1].get_input_size().height, model[layer_index-1].get_input_size().width);
    }
}