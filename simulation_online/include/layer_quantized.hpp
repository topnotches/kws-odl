#ifndef LAYER_QUANTIZED_H
#define LAYER_QUANTIZED_H
#include <stdint.h>
#include <vector>
#include <string>
#include "defs.hpp"
class layer_q {
private:
    double layer_rescale_value;
    double layer_bw_rescale_value;
    quant_param_t layer_quant_params;
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
    float layer_adam_learning_rate = 1e-5;
    float layer_adam_beta_scale;
    const uint8_t layer_adam_beta_scale_shifts = 7;
    const int32_t layer_adam_beta1 = 115; // 0.9 I think
    const int32_t layer_adam_beta2 = 127; // 0.99 I think
    std::vector<int32_t> layer_adam_momentum;
    std::vector<int32_t> layer_adam_velocity;
    float layer_sgd_learning_rate = 1; // Just a bit above 1e-5 for 18 bits
    float layer_adam_epsilon = 1e-8;
    uint32_t layer_adam_time_step = 1;
    float layer_sgd_momentum_scale;
    const uint8_t layer_sgd_momentum_scale_shifts = 7;
    const int32_t layer_sgd_momentum = 127; // 0.99 I think
    std::vector<int32_t> layer_sgd_velocity;
public:
    std::vector<float> debug_float;
    std::vector<int32_t> debug_fixed;
    std::vector<int32_t> layer_gradient_outputs;
    std::vector<int32_t> layer_inputs;
    std::vector<int32_t> layer_outputs;
    double get_rescale_value();
    double get_bw_rescale_value();
    tensor_dim_sizes_t get_input_size();
    tensor_dim_sizes_t get_output_size();
    quant_param_t get_qparams();
    std::vector<int32_t> get_weights();
    std::vector<int32_t> get_biases();
    void set_dense_bw_output_scale(float scale);
    std::vector<int32_t> get_layer_bn_means();
    std::vector<int32_t> get_layer_bn_variances();
    LayerTypes get_layer_type();
    layer_q(LayerTypes            layer_type, 
          tensor_dim_sizes_t    layer_dim_size_in, 
          quant_param_t         layer_quant_params = {},
          int32_t                 *weights = nullptr, 
          int32_t                 *biases = nullptr, 
          conv_hypr_param_t     layer_conv_hypr_params = {},
          dense_hypr_param_t    layer_dense_hypr_params = {},
          int32_t                 *layer_bn_means = NULL,
          int32_t                 *layer_bn_variances = NULL);
    ~layer_q();
    void forward(int32_t *layer_input, int32_t *labels_input = NULL);
    void backward(int32_t *layer_gradient_input);
    void adam_optimize(const int32_t* layer_adam_gradients_backprop, const uint32_t layer_adam_size);
    void stochastic_gradient_descent_optimize(const int32_t* layer_stochastic_gradient_descent_gradients_backprop, const uint32_t layer_stochastic_gradient_descent_size);
    void print_layer_type();
};


#endif