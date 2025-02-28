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
    // Adam stuff:
    float layer_adam_learning_rate = 1e-5;
    float layer_adam_beta1 = 0.9;
    float layer_adam_beta2 = 0.99;
    std::vector<float> layer_adam_momentum;
    std::vector<float> layer_adam_velocity;
    float layer_adam_epsilon = 1e-8;
    uint32_t layer_adam_time_step = 1;
    public:
    std::vector<float> layer_gradient_outputs;
    std::vector<float> layer_inputs;
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
    void forward(float *layer_input, float *labels_input = NULL);
    void backward(float *layer_gradient_input);
    void adam_optimize(const float* layer_adam_gradients_backprop, const uint32_t layer_adam_size);
    void print_layer_type();
};


#endif