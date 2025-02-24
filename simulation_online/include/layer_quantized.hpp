#ifndef LAYER_QUANTIZED_H
#define LAYER_QUANTIZED_H
#include <stdint.h>
#include <vector>
#include <string>
#include "defs.hpp"
class layer_q {
private:
    float layer_rescale_value;
    std::vector<int32_t> layer_weights;
    std::vector<int32_t> layer_biases;
    tensor_dim_sizes_t layer_dim_size_in;  // {width, height, depth, batch_size}
    tensor_dim_sizes_t layer_dim_size_out; // {width, height, depth, batch_size}
    conv_hypr_param_t layer_conv_hypr_params;
    dense_hypr_param_t layer_dense_hypr_params;
    std::vector<int32_t> layer_bn_means;
    std::vector<int32_t> layer_bn_variances;
    LayerTypes layer_type;
    // Adam stuff:
    int32_t layer_adam_learning_rate = 1e-5;
    int32_t layer_adam_beta1 = 0.9;
    int32_t layer_adam_beta2 = 0.99;
    std::vector<int32_t> layer_adam_momentum;
    std::vector<int32_t> layer_adam_velocity;
    int32_t layer_adam_epsilon = 1e-8;
    uint32_t layer_adam_time_step = 1;
public:
std::vector<int32_t> layer_gradient_outputs;
    std::vector<int32_t> layer_outputs;
    tensor_dim_sizes_t get_input_size();
    tensor_dim_sizes_t get_output_size();
    std::vector<int32_t> get_weights();
    std::vector<int32_t> get_biases();
    std::vector<int32_t> get_layer_bn_means();
    std::vector<int32_t> get_layer_bn_variances();
    LayerTypes get_layer_type();
    layer_q(LayerTypes            layer_type, 
          tensor_dim_sizes_t    layer_dim_size_in, 
          int32_t                 *weights = nullptr, 
          int32_t                 *biases = nullptr, 
          quant_param_t         layer_quant_params = {},
          conv_hypr_param_t     layer_conv_hypr_params = {},
          dense_hypr_param_t    layer_dense_hypr_params = {},
          int32_t                 *layer_bn_means = NULL,
          int32_t                 *layer_bn_variances = NULL);
    ~layer_q();
    void forward(int32_t *layer_input, int32_t *labels_input = NULL);
    void backward(int32_t *layer_gradient_input);
    void adam_optimize(const int32_t* layer_adam_gradients_backprop, const uint32_t layer_adam_size);
    void print_layer_type();
};


#endif