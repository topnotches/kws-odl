#include <iostream>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <vector>
#include "misc_utils.hpp"
#include "defs.hpp"
#include "layer.hpp"

#define BATCH_SIZE 2

std::vector<float> load_mffcs_bin(const std::string& filename) {
    std::vector<float> data;
    
    // Open the binary file for reading
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    // Get the file size
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Ensure the file size is a multiple of float size
    if (size % sizeof(float) != 0) {
        std::cerr << "File size is not a multiple of float size!" << std::endl;
        return data;
    }

    // Resize the vector to hold the float data
    size_t numFloats = size / sizeof(float);
    data.resize(numFloats);

    // Read the binary file into the float vector
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        std::cerr << "Error reading file: " << filename << std::endl;
        data.clear();
    }

    return data;
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

dense_hypr_param_t set_dense_param(uint32_t output_size, uint32_t input_size = 1) {
    dense_hypr_param_t params;

    params.size_in = input_size;
    params.size_out = output_size;
    
    return params;
}


int  main() {
    std::string exported_params = "./exported_models/export_params_nclass_12.csv";
    std::string test_mfccs_path = "../dataset_mfccs_raw/yes/d21fd169_nohash_0";
    std::vector<std::vector<std::string>> layer_params_str = load_layers_from_csv_to_vec(exported_params);

    std::vector<std::vector<float>> layer_params;
    uint8_t i_print = 0;
    for (auto layer_name: layer_params_str){
        layer_params.push_back(str_to_fl32_vec(layer_name[2]));
        //std::cout << "index: " << std::to_string(i_print) << ", name: " << layer_name[0].c_str() << std::endl;
        //i_print ++;
    }
    // Model is list of layers
    std::vector<layer> model;
    // MFCC input dimension

    tensor_dim_sizes_t layer_dim_size_in;

    layer_dim_size_in.height = 49;
    layer_dim_size_in.width = 10;
    layer_dim_size_in.depth = 1;
    layer_dim_size_in.batch = 1;
    layer_dim_size_in.full = layer_dim_size_in.width *
                            layer_dim_size_in.height *
                            layer_dim_size_in.depth *
                            layer_dim_size_in.batch;

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
    dense_hypr_param_t param_dense_layer7 = set_dense_param(NUMBER_OF_CLASSES);

    /****************************************
    ***** Create Instantiate Each Layer *****
    ****************************************/
    // Layer 1, Convolution
    model.push_back(layer(LayerTypes::conv,         layer_dim_size_in,               layer_params[0].data(),  layer_params[1].data(), param_conv_layer1));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[2].data(),  layer_params[3].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    // Layer 2, Depth-wise Separable Convolution
    model.push_back(layer(LayerTypes::dw_conv,      model.back().get_output_size(),  layer_params[4].data(),  layer_params[5].data(), param_dw_conv_layer2));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[6].data(),  layer_params[7].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    model.push_back(layer(LayerTypes::conv,         model.back().get_output_size(),  layer_params[8].data(),  layer_params[9].data(), param_pw_conv_layer2));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[10].data(), layer_params[11].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));

    // Layer 3, Depth-wise Separable Convolution
    model.push_back(layer(LayerTypes::dw_conv,      model.back().get_output_size(),  layer_params[12].data(), layer_params[13].data(), param_dw_conv_layer2));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[14].data(),  layer_params[15].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    model.push_back(layer(LayerTypes::conv,         model.back().get_output_size(),  layer_params[16].data(),  layer_params[17].data(), param_pw_conv_layer2));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[18].data(),  layer_params[19].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));

    // Layer 4, Depth-wise Separable Convolution
    model.push_back(layer(LayerTypes::dw_conv,      model.back().get_output_size(),  layer_params[20].data(),  layer_params[21].data(), param_dw_conv_layer2));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[22].data(),  layer_params[23].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    model.push_back(layer(LayerTypes::conv,         model.back().get_output_size(),  layer_params[24].data(),  layer_params[25].data(), param_pw_conv_layer2));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[26].data(),  layer_params[27].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));

    // Layer 5, Depth-wise Separable Convolution
    model.push_back(layer(LayerTypes::dw_conv,      model.back().get_output_size(),  layer_params[28].data(),  layer_params[29].data(), param_dw_conv_layer2));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[30].data(),  layer_params[31].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));
    model.push_back(layer(LayerTypes::conv,         model.back().get_output_size(),  layer_params[32].data(),  layer_params[33].data(), param_pw_conv_layer2));
    model.push_back(layer(LayerTypes::batchnorm,    model.back().get_output_size(),  layer_params[34].data(),  layer_params[35].data()));
    model.push_back(layer(LayerTypes::relu,         model.back().get_output_size()));

    // Layer 6, Average Pooling
    model.push_back(layer(LayerTypes::avgpool2d,    model.back().get_output_size(), {}, {}, param_ap2d_conv_layer6));

    // Layer 7, Dense 
    model.push_back(layer(LayerTypes::dense,        model.back().get_output_size(), layer_params[36].data(),  layer_params[37].data(), {}, param_dense_layer7));
    std::vector<float> input_mfccs = load_mffcs_bin(test_mfccs_path);
    
    model[0].forward(input_mfccs.data());

    for (uint8_t layer_index = 1; layer_index < model.size(); layer_index++ ) {

       model[layer_index].forward(model[layer_index - 1].layer_outputs.data());
    }
    std::vector<float> weights =  model[0].get_weights();
    for (auto activation : weights) {
        std::cout << weights.size() << std::endl;
    }
    return 0;
}

/*


    for (auto my_layer : model) {
       auto my_size = my_layer.get_output_size(); 
       std::cout << "[" <<
       my_size.batch << " " <<
       my_size.depth << " " <<
       my_size.height << " " <<
       my_size.width << " " <<
       "]" <<
       std::endl;
    }
     ConstantPad2d-1            [-1, 1, 59, 12]               0
            Conv2d-2            [-1, 64, 25, 5]           2,624
       BatchNorm2d-3            [-1, 64, 25, 5]             128
              ReLU-4            [-1, 64, 25, 5]               0

     ConstantPad2d-5            [-1, 64, 27, 7]               0
            Conv2d-6            [-1, 64, 25, 5]             640
       BatchNorm2d-7            [-1, 64, 25, 5]             128
              ReLU-8            [-1, 64, 25, 5]               0
            Conv2d-9            [-1, 64, 25, 5]           4,160
      BatchNorm2d-10            [-1, 64, 25, 5]             128
             ReLU-11            [-1, 64, 25, 5]               0

    ConstantPad2d-12            [-1, 64, 27, 7]               0
           Conv2d-13            [-1, 64, 25, 5]             640
      BatchNorm2d-14            [-1, 64, 25, 5]             128
             ReLU-15            [-1, 64, 25, 5]               0
           Conv2d-16            [-1, 64, 25, 5]           4,160
      BatchNorm2d-17            [-1, 64, 25, 5]             128
             ReLU-18            [-1, 64, 25, 5]               0

    ConstantPad2d-19            [-1, 64, 27, 7]               0
           Conv2d-20            [-1, 64, 25, 5]             640
      BatchNorm2d-21            [-1, 64, 25, 5]             128
             ReLU-22            [-1, 64, 25, 5]               0
           Conv2d-23            [-1, 64, 25, 5]           4,160
      BatchNorm2d-24            [-1, 64, 25, 5]             128
             ReLU-25            [-1, 64, 25, 5]               0

    ConstantPad2d-26            [-1, 64, 27, 7]               0
           Conv2d-27            [-1, 64, 25, 5]             640
      BatchNorm2d-28            [-1, 64, 25, 5]             128
             ReLU-29            [-1, 64, 25, 5]               0
           Conv2d-30            [-1, 64, 25, 5]           4,160
      BatchNorm2d-31            [-1, 64, 25, 5]             128
             ReLU-32            [-1, 64, 25, 5]               0

        AvgPool2d-33             [-1, 64, 1, 1]               0
           Linear-34                   [-1, 12]             780
           */