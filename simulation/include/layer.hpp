#ifndef LAYER_H
#define LAYER_H
#include <stdint.h>
#include <vector>
#include <string>
#include "defs.hpp"
class layer {
private:
    std::vector<float> layer_weights; 
    std::vector<float> layer_biases;
    tensor_dim_sizes_t layer_dim_size_in;  // {width, height, depth, batch_size}
    tensor_dim_sizes_t layer_dim_size_out; // {width, height, depth, batch_size}
    conv_hypr_param_t layer_conv_hypr_params;
    dense_hypr_param_t layer_dense_hypr_params;
    std::vector<float> layer_bn_means;
    std::vector<float> layer_bn_variances;
    LayerTypes layer_type;
public:
    std::vector<float> layer_outputs;
    tensor_dim_sizes_t get_input_size();
    tensor_dim_sizes_t get_output_size();
    std::vector<float> get_weights();
    std::vector<float> get_biases();
    std::vector<float> get_layer_bn_means();
    std::vector<float> get_layer_bn_variances();
    LayerTypes get_layer_type();
    layer(LayerTypes            layer_type, 
          tensor_dim_sizes_t    layer_dim_size_in, 
          float                 *weights = nullptr, 
          float                 *biases = nullptr, 
          conv_hypr_param_t     layer_conv_hypr_params = {},
          dense_hypr_param_t    layer_dense_hypr_params = {},
          float                 *layer_bn_means = NULL,
          float                 *layer_bn_variances = NULL);
    ~layer();
    void forward(float *layer_input);
};


#endif