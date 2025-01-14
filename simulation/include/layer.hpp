#ifndef LAYER_H
#define LAYER_H
#include <stdint.h>
#include <vector>
#include <string>
#include "defs.hpp"
class layer
{
private:
    std::vector<float> layer_weights; 
    std::vector<float> layer_biases;
    tensor_dim_sizes_t layer_dim_size_in;  // {width, height, depth, batch_size}
    tensor_dim_sizes_t layer_dim_size_out; // {width, height, depth, batch_size}
    conv_hypr_param_t layer_conv_hypr_params;
    dense_param_t layer_dense_conv_hypr_params;
    LayerTypes layer_type;
public:
    std::vector<float> layer_outputs;
    layer(LayerTypes layer_type, tensor_dim_sizes_t layer_dim_size_in,
    float *weights, float *biases, conv_hypr_param_t conv_params);
    ~layer();
};


#endif