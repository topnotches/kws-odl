#include "layer_quantized.hpp"
#include "defs.hpp"
#include <iostream>
#include <math.h>
#include <random>
#include "misc_utils.hpp"
#include "quantization_utils.hpp"

#include "batch_norm_layer.hpp"
#include "conv_layer.hpp"
#include "dense_layer.hpp"
#include "dw_conv_layer.hpp"
#include "relu_layer.hpp"
#include "avgpool2d_layer.hpp"
#include "softmax_layer.hpp"
#include "cross_entropy_loss_layer.hpp"
#include "fusion_layer.hpp"


layer_q::layer_q(LayerTypes     layer_type, 
          tensor_dim_sizes_t    layer_dim_size_in, 
          quant_param_t         layer_quant_params,
          int32_t                *weights, 
          int32_t                *biases,
          conv_hypr_param_t     layer_conv_hypr_params,
          dense_hypr_param_t    layer_dense_hypr_params,
          int32_t                *layer_bn_means,
          int32_t                *layer_bn_variances) {

    this->layer_weights = {};
    this->layer_biases = {};
    this->layer_dim_size_in = {};  // {width, height, depth, batch_size}
    this->layer_dim_size_out = {}; // {width, height, depth, batch_size}
    this->layer_conv_hypr_params = {};
    this->layer_dense_hypr_params = {};
    this->layer_outputs.resize(0);
    this->layer_gradient_outputs.resize(0);
    this->layer_rescale_value = (layer_quant_params.scale_weight*layer_quant_params.scale_in)/layer_quant_params.scale_out;
    this->layer_bw_rescale_value = (layer_quant_params.scale_weight*layer_quant_params.scale_in)/layer_quant_params.scale_out;
    this->layer_quant_params = layer_quant_params;
    this->layer_adam_beta_scale = 1/((int)pow(2,this->layer_adam_beta_scale_shifts)+0.5);
    this->layer_sgd_momentum_scale = 1/((int)pow(2,this->layer_sgd_momentum_scale_shifts)+0.5);
    this->layer_dim_size_in = layer_dim_size_in; // {width, height, depth, batch_size}

    switch (layer_type) { 
        case LayerTypes::conv: {
            this->layer_type = LayerTypes::conv; 
                    this->layer_conv_hypr_params = layer_conv_hypr_params;

            this->layer_dim_size_out.width  = (this->layer_dim_size_in.width - this->layer_conv_hypr_params.kernel_width + this->layer_conv_hypr_params.pad_left + this->layer_conv_hypr_params.pad_right) / this->layer_conv_hypr_params.kernel_stride + 1;
            this->layer_dim_size_out.height = (this->layer_dim_size_in.height - this->layer_conv_hypr_params.kernel_height + this->layer_conv_hypr_params.pad_top + this->layer_conv_hypr_params.pad_bottom) / this->layer_conv_hypr_params.kernel_stride + 1;
            this->layer_dim_size_out.depth  = this->layer_conv_hypr_params.kernel_count;
            this->layer_dim_size_out.batch  = this->layer_dim_size_in.batch;
            this->layer_dim_size_out.full = this->layer_dim_size_out.width *
                                            this->layer_dim_size_out.height *
                                            this->layer_dim_size_out.depth *
                                            this->layer_dim_size_out.batch;
            uint32_t layer_conv_weight_count = this->layer_conv_hypr_params.kernel_width *
                                            this->layer_conv_hypr_params.kernel_height *
                                            this->layer_conv_hypr_params.kernel_count *
                                            this->layer_dim_size_in.depth;

            uint32_t layer_conv_bias_count = this->layer_conv_hypr_params.kernel_count;
                    this->layer_weights.resize(0);
            this->layer_weights.insert(this->layer_weights.end(), weights, weights + layer_conv_weight_count);

            this->layer_biases.resize(0);
            this->layer_biases.insert(this->layer_biases.end(), biases, biases + layer_conv_bias_count);

            this->layer_outputs.resize(this->layer_dim_size_out.full);
            this->layer_gradient_outputs.resize(this->layer_dim_size_in.full);

            break;
        }
        case LayerTypes::dw_conv: {
            this->layer_type = LayerTypes::dw_conv;

            this->layer_conv_hypr_params = layer_conv_hypr_params;
            this->layer_dim_size_out.width = (this->layer_dim_size_in.width - this->layer_conv_hypr_params.kernel_width + this->layer_conv_hypr_params.pad_left + this->layer_conv_hypr_params.pad_right) / this->layer_conv_hypr_params.kernel_stride + 1;
            this->layer_dim_size_out.height = (this->layer_dim_size_in.height - this->layer_conv_hypr_params.kernel_height + this->layer_conv_hypr_params.pad_top + this->layer_conv_hypr_params.pad_bottom) / this->layer_conv_hypr_params.kernel_stride + 1;
            this->layer_dim_size_out.depth  = this->layer_dim_size_in.depth;
            this->layer_dim_size_out.batch  = this->layer_dim_size_in.batch;
            this->layer_dim_size_out.full = this->layer_dim_size_out.width *
                                    this->layer_dim_size_out.height *
                                    this->layer_dim_size_out.depth *
                                    this->layer_dim_size_out.batch;
            uint32_t layer_conv_weight_count = this->layer_conv_hypr_params.kernel_width *
                                        this->layer_conv_hypr_params.kernel_height *
                                        this->layer_dim_size_in.depth;
            uint32_t layer_conv_bias_count = this->layer_dim_size_in.depth;
            this->layer_weights.resize(0);
            this->layer_weights.insert(this->layer_weights.end(), weights, weights + layer_conv_weight_count);

            this->layer_biases.resize(0);
            this->layer_biases.insert(this->layer_biases.end(), biases, biases + layer_conv_bias_count);

            this->layer_outputs.resize(this->layer_dim_size_out.full);
            this->layer_gradient_outputs.resize(this->layer_dim_size_in.full);

            break;
        }
        case LayerTypes::dense: {
            this->layer_type = LayerTypes::dense;
            this->layer_bw_rescale_value = (layer_quant_params.scale_weight*(1.0f/255.0f))/DENSE_BW_OUTPUT_SCALE;

            //(0.008598939515650272*(1/256)
            this->layer_dense_hypr_params.size_in = this->layer_dim_size_in.full / this->layer_dim_size_in.batch;
            this->layer_dense_hypr_params.size_out = layer_dense_hypr_params.size_out;
                    this->layer_dim_size_out.full = this->layer_dense_hypr_params.size_out * this->layer_dim_size_in.batch;
            this->layer_dim_size_out.width = this->layer_dense_hypr_params.size_out; // change and test for change == 1
            this->layer_dim_size_out.height = 1;
            this->layer_dim_size_out.depth = 1;
            this->layer_dim_size_out.batch = this->layer_dim_size_in.batch;

            uint32_t layer_dense_weight_count = this->layer_dense_hypr_params.size_in * this->layer_dense_hypr_params.size_out;
            uint32_t layer_dense_bias_count = this->layer_dense_hypr_params.size_out;
                    this->layer_weights.resize(0);
            this->layer_weights.insert(this->layer_weights.end(), weights, weights + layer_dense_weight_count);

            this->layer_biases.resize(0);
            this->layer_biases.insert(this->layer_biases.end(), biases, biases + layer_dense_bias_count);

            this->layer_outputs.resize(this->layer_dim_size_out.full);
            this->layer_gradient_outputs.resize(this->layer_dim_size_in.full);

            break;
        }
        case LayerTypes::batchnorm: {

            this->layer_type = LayerTypes::batchnorm;
            this->layer_dim_size_out = this->layer_dim_size_in;
            uint32_t layer_bn_weight_count = this->layer_dim_size_in.depth;
            uint32_t layer_bn_bias_count = this->layer_dim_size_in.depth;
            this->layer_weights.resize(0);
            this->layer_weights.insert(this->layer_weights.end(), weights, weights + layer_bn_weight_count);

            this->layer_biases.resize(0);
            this->layer_biases.insert(this->layer_biases.end(), biases, biases + layer_bn_bias_count);
            uint32_t layer_bn_mean_count = this->layer_dim_size_in.depth;
            uint32_t layer_bn_variance_count = this->layer_dim_size_in.depth;
            this->layer_bn_means.resize(0);
            this->layer_bn_means.insert(this->layer_bn_means.end(), layer_bn_means, layer_bn_means + layer_bn_mean_count);

            this->layer_bn_variances.resize(0);
            this->layer_bn_variances.insert(this->layer_bn_variances.end(), layer_bn_variances, layer_bn_variances + layer_bn_variance_count);

            this->layer_outputs.resize(this->layer_dim_size_out.full);
            this->layer_gradient_outputs.resize(this->layer_dim_size_in.full);

            break;
          }  
        case LayerTypes::relu: {
            this->layer_type = LayerTypes::relu;

            this->layer_dim_size_out = this->layer_dim_size_in;

            this->layer_outputs.resize(this->layer_dim_size_out.full);
            this->layer_gradient_outputs.resize(this->layer_dim_size_in.full);

            break;
        }
        case LayerTypes::avgpool2d: {
            this->layer_type = LayerTypes::avgpool2d;

            this->layer_conv_hypr_params = layer_conv_hypr_params;

            this->layer_dim_size_out.width = (this->layer_dim_size_in.width - this->layer_conv_hypr_params.kernel_width + this->layer_conv_hypr_params.pad_left + this->layer_conv_hypr_params.pad_right) / this->layer_conv_hypr_params.kernel_stride + 1;
            this->layer_dim_size_out.height = (this->layer_dim_size_in.height - this->layer_conv_hypr_params.kernel_height + this->layer_conv_hypr_params.pad_top + this->layer_conv_hypr_params.pad_bottom) / this->layer_conv_hypr_params.kernel_stride + 1;
            this->layer_dim_size_out.depth  = this->layer_dim_size_in.depth;
            this->layer_dim_size_out.batch  = this->layer_dim_size_in.batch;
            this->layer_dim_size_out.full = this->layer_dim_size_out.width *
                                    this->layer_dim_size_out.height *
                                    this->layer_dim_size_out.depth *
                                    this->layer_dim_size_out.batch;
            this->layer_outputs.resize(this->layer_dim_size_out.full);
            this->layer_gradient_outputs.resize(this->layer_dim_size_in.full);

            break;
        }
        case LayerTypes::softmax: {
            this->layer_type = LayerTypes::softmax;
                    this->layer_dim_size_out = this->layer_dim_size_in;
                    this->layer_outputs.resize(this->layer_dim_size_out.full);
            this->layer_gradient_outputs.resize(this->layer_dim_size_in.full);
            //std::cout << this->layer_dim_size_in.full<< std::endl;

            break;
        }
        case LayerTypes::cross_entropy_loss: {
            this->layer_type = LayerTypes::cross_entropy_loss;
            this->layer_dim_size_out.batch = this->layer_dim_size_in.batch; 
            this->layer_dim_size_out.full = this->layer_dim_size_in.batch; 
            this->layer_dim_size_out.depth = 1; 
            this->layer_dim_size_out.height = 1; 
            this->layer_dim_size_out.width = 1; 

            this->layer_outputs.resize(this->layer_dim_size_out.full);
            this->layer_gradient_outputs.resize(this->layer_dim_size_in.full);

            break;
        }
        case LayerTypes::fusion: {

#if DO_ADDITION_ELSE_WEIGHT_FUSION
            this->layer_bw_rescale_value = (DENSE_BW_OUTPUT_SCALE)/layer_quant_params.scale_weight;            
#else
            this->layer_bw_rescale_value = (layer_quant_params.scale_in*DENSE_BW_OUTPUT_SCALE)/layer_quant_params.scale_weight;
#endif
            this->layer_type = LayerTypes::fusion;
            this->layer_dim_size_out = this->layer_dim_size_in; 
            srand((unsigned)time(NULL));
            this->layer_outputs.resize(this->layer_dim_size_out.full);
            this->layer_dim_size_out.width = this->layer_dim_size_out.full / this->layer_dim_size_out.batch;
            this->layer_weights.resize(this->layer_dim_size_out.width);
#if DO_ADDITION_ELSE_WEIGHT_FUSION
            std::fill(this->layer_weights.begin(), this->layer_weights.end(), 0); // Fill with zeros if multiplication fusion
#else
            std::fill(this->layer_weights.begin(), this->layer_weights.end(), (int)(pow(2, this->layer_quant_params.weight_bits-1) + 0.5)); // Fill with ones if multiplication fusion
#endif
            this->layer_adam_momentum.resize(this->layer_weights.size());
            std::fill(this->layer_adam_momentum.begin(), this->layer_adam_momentum.end(), 0); 

            this->layer_adam_velocity.resize(this->layer_weights.size());
            std::fill(this->layer_adam_velocity.begin(), this->layer_adam_velocity.end(), 0); 

            this->layer_sgd_velocity.resize(this->layer_weights.size());
            std::fill(this->layer_sgd_velocity.begin(), this->layer_sgd_velocity.end(), 0); 

            this->layer_gradient_outputs.resize(this->layer_dim_size_in.full);


            break;
        }
        default: {
            KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("Trying to initialize an unsupported layer...");
            break;
        }  
    }
}

layer_q::~layer_q() {

}


void layer_q::forward(int32_t *layer_input, int32_t *labels_input) {
    switch (this->layer_type) {
        case LayerTypes::conv: {
            conv_layer_fixed(layer_input, this->layer_outputs.data(), this->layer_weights.data(), this->layer_biases.data(),
                            this->layer_dim_size_in.width, this->layer_dim_size_in.height, this->layer_dim_size_in.depth,
                            this->layer_conv_hypr_params.kernel_stride, this->layer_conv_hypr_params.kernel_width, this->layer_conv_hypr_params.kernel_height,
                            this->layer_conv_hypr_params.kernel_count, this->layer_dim_size_out.batch,
                            this->layer_conv_hypr_params.pad_top, this->layer_conv_hypr_params.pad_bottom,
                            this->layer_conv_hypr_params.pad_left, this->layer_conv_hypr_params.pad_right, this->layer_rescale_value, this->layer_quant_params.activation_bits);
            break;
        }
        case LayerTypes::dw_conv: {
            dw_conv_layer_fixed(layer_input, this->layer_outputs.data(), this->layer_weights.data(), this->layer_biases.data(),
                                        this->layer_dim_size_in.width, this->layer_dim_size_in.height, this->layer_dim_size_in.depth,
                                        this->layer_conv_hypr_params.kernel_stride, this->layer_conv_hypr_params.kernel_width,
                                        this->layer_conv_hypr_params.kernel_height, this->layer_dim_size_in.batch,
                                        this->layer_conv_hypr_params.pad_top, this->layer_conv_hypr_params.pad_bottom,
                                        this->layer_conv_hypr_params.pad_left, this->layer_conv_hypr_params.pad_right, this->layer_rescale_value, this->layer_quant_params.activation_bits);

            break;
        }
        case LayerTypes::dense: {
            
            dense_layer_fixed(layer_input, this->layer_outputs.data(), this->layer_weights.data(), this->layer_biases.data(),
                        this->layer_dense_hypr_params.size_in, this->layer_dense_hypr_params.size_out, this->layer_dim_size_out.batch, this->layer_rescale_value, this->layer_quant_params.activation_bits);

            break;
        }
        case LayerTypes::batchnorm: {

            KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("Trying forward() on BATCHNORM...");
            //batch_norm_fixed(layer_input, this->layer_outputs.data(), this->layer_weights.data(), this->layer_biases.data(),
            //                        this->layer_bn_means.data(), this->layer_bn_variances.data(),
            //                        this->layer_dim_size_in.width * this->layer_dim_size_in.height, this->layer_dim_size_in.depth, this->layer_dim_size_in.batch, this->layer_rescale_value, this->layer_quant_params.activation_bits);

            break;
          }  
        case LayerTypes::relu: {

            relu_layer_fixed(layer_input, this->layer_outputs.data(), this->layer_dim_size_in.width, this->layer_dim_size_in.height, this->layer_dim_size_in.depth*this->layer_dim_size_in.batch, this->layer_rescale_value, this->layer_quant_params.activation_bits);
            break;
        }
        case LayerTypes::avgpool2d: {
            avgpool2d_layer_fixed(layer_input, this->layer_outputs.data(),
                                        this->layer_dim_size_in.width, this->layer_dim_size_in.height, this->layer_dim_size_in.depth,
                                        this->layer_conv_hypr_params.kernel_stride, this->layer_conv_hypr_params.kernel_width,
                                        this->layer_conv_hypr_params.kernel_height, this->layer_dim_size_in.batch,
                                        this->layer_conv_hypr_params.pad_top, this->layer_conv_hypr_params.pad_bottom,
                                        this->layer_conv_hypr_params.pad_left, this->layer_conv_hypr_params.pad_right, this->layer_rescale_value, this->layer_quant_params.activation_bits);
            break;
        }
        case LayerTypes::softmax: {
            softmax_layer_fixed(layer_input, this->layer_outputs.data(), this->layer_dim_size_in.batch, this->layer_dim_size_in.width, this->layer_rescale_value, this->layer_quant_params.activation_bits);
            break;
        }
        case LayerTypes::cross_entropy_loss: {
            cross_entropy_loss_fixed(labels_input, layer_input, this->layer_outputs.data(),  this->layer_dim_size_out.batch, this->layer_dim_size_in.width);
            break;
        }
        case LayerTypes::fusion: {
#if DO_ADDITION_ELSE_WEIGHT_FUSION
            fusion_add_fixed(layer_input, this->layer_outputs.data(), this->layer_weights.data(), this->layer_dim_size_out.width, this->layer_dim_size_out.batch, this->layer_rescale_value, this->layer_quant_params.activation_bits);
#else
            fusion_mult_fixed(layer_input, this->layer_outputs.data(), this->layer_weights.data(), this->layer_dim_size_out.width, this->layer_dim_size_out.batch, this->layer_rescale_value, this->layer_quant_params.activation_bits);
            for (uint32_t i = 0; i < this->layer_dim_size_in.full; i ++) {
                this->layer_inputs.push_back(layer_input[i]);
            }
#endif


            break;
        }
        default: {
            KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("Trying forward() on an unsupported layer...");
            break;
        }  
    }
}

void layer_q::backward(int32_t *layer_gradient_input) {
    switch (this->layer_type) {
        case LayerTypes::conv: {
            KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("Trying backward() on an unsupported layer...");
            break;
        }
        case LayerTypes::dw_conv: {
            KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("Trying backward() on an unsupported layer...");
            break;
        }
        case LayerTypes::dense: {
            dense_layer_backward_fixed(
                             this->layer_gradient_outputs.data(), 
                                layer_gradient_input,
                                this->layer_weights.data(),
                                this->layer_dense_hypr_params.size_in, 
                                this->layer_dense_hypr_params.size_out, 
                                this->layer_dim_size_in.batch,
                                this->layer_bw_rescale_value,
                                this->layer_quant_params.gradient_bits);
                                            
            break;
        }
        case LayerTypes::batchnorm: {
            KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("Trying backward() on an unsupported layer...");
            break;
          }  
        case LayerTypes::relu: {
            KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("Trying backward() on an unsupported layer...");
            break;

        }
        case LayerTypes::avgpool2d: {
            KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("Trying backward() on an unsupported layer...");
            break;
        }
        case LayerTypes::softmax: {
            softmax_layer_backward_fixed(this->layer_outputs.data(), 
                                        layer_gradient_input, 
                                        this->layer_gradient_outputs.data(), 
                                        this->layer_dim_size_out.batch, 
                                        this->layer_dim_size_out.width,
                                        this->layer_bw_rescale_value,
                                        this->layer_quant_params.gradient_bits);
            break;
        }
        case LayerTypes::cross_entropy_loss: {
            KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("Trying to backward() cross_entropy layer which is coupled with softmax... Use softmax instead!");
            break;
        }
        case LayerTypes::fusion: {

#if DO_ADDITION_ELSE_WEIGHT_FUSION
            fusion_add_backward_fixed(this->layer_gradient_outputs.data(), 
                        layer_gradient_input, 
                        this->layer_inputs.data(),
                        this->layer_dim_size_out.width, 
                        this->layer_dim_size_out.batch,
                        this->layer_bw_rescale_value,
                        this->layer_quant_params.gradient_bits);
#else
            fusion_mult_backward_fixed(this->layer_gradient_outputs.data(), 
                        layer_gradient_input, 
                        this->layer_inputs.data(),
                        this->layer_dim_size_out.width, 
                        this->layer_dim_size_out.batch,
                        this->layer_bw_rescale_value,
                        this->layer_quant_params.gradient_bits);
#endif
                

#if DO_SGD_ELSE_ADAM
        stochastic_gradient_descent_optimize(this->layer_gradient_outputs.data(), this->layer_dim_size_out.width);
#else
        adam_optimize(this->layer_gradient_outputs.data(), this->layer_dim_size_out.width);
#endif
            break;
        }
        default: {
            KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("Trying to backward() a layer that doesn't have a type...");
            break;
        }  
    }
}
void layer_q::print_layer_type() {
    switch (this->layer_type) {
        case LayerTypes::conv: {
            std::cout << "LAYER_TYPE: " << "conv" << std::endl;
            break;
        }
        case LayerTypes::dw_conv: {
            std::cout << "LAYER_TYPE: " << "dw_conv" << std::endl;
            break;
        }
        case LayerTypes::dense: {
            std::cout << "LAYER_TYPE: " << "dense" << std::endl;
            break;
        }
        case LayerTypes::batchnorm: {
            std::cout << "LAYER_TYPE: " << "batchnorm" << std::endl;
            break;
          }  
        case LayerTypes::relu: {
            std::cout << "LAYER_TYPE: " << "relu" << std::endl;
            break;

        }
        case LayerTypes::avgpool2d: {
            std::cout << "LAYER_TYPE: " << "avgpool2d" << std::endl;
            break;
        }
        case LayerTypes::softmax: {
            std::cout << "LAYER_TYPE: " << "softmax" << std::endl;
            break;
        }
        case LayerTypes::cross_entropy_loss: {
            std::cout << "LAYER_TYPE: " << "cross_entropy_loss" << std::endl;
            break;
        }
        case LayerTypes::fusion: {
            std::cout << "LAYER_TYPE: " << "fusion" << std::endl;
            break;
        }
        default: {
            KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("CANT PRINT NON-EXISTENT LAYER!");
            break;
        }  
    }
}
void layer_q::adam_optimize(const int32_t* layer_adam_gradients_backprop, const uint32_t layer_adam_size) {
    std::vector<int32_t> avg_gradients(layer_adam_size, 0);

    for (uint32_t feature_index = 0; feature_index < layer_adam_size; feature_index++) {
        int32_t sum = 0;
        for (uint32_t batch_index = 0; batch_index < this->layer_dim_size_in.batch; batch_index++) {
            sum += layer_adam_gradients_backprop[feature_index * this->layer_dim_size_in.batch + batch_index];
        }
        avg_gradients[feature_index] = sum / this->layer_dim_size_in.batch;

    }


    float beta1_f = static_cast<float>(this->layer_adam_beta1) / 128.0f;
    float beta2_f = static_cast<float>(this->layer_adam_beta2) / 128.0f;
    float bias_correction1 = 1.0f - pow(beta1_f, static_cast<float>(this->layer_adam_time_step));
    float bias_correction2 = 1.0f - pow(beta2_f, static_cast<float>(this->layer_adam_time_step));

    for (uint32_t index = 0; index < layer_adam_size; index++) {

        this->layer_adam_momentum[index] = requantize_shift(this->layer_adam_beta1 * this->layer_adam_momentum[index], this->layer_adam_beta_scale, this->layer_quant_params.weight_bits, false) + requantize_shift((128 - this->layer_adam_beta1) * avg_gradients[index], this->layer_adam_beta_scale, this->layer_quant_params.weight_bits, false);
        this->layer_adam_velocity[index] = requantize_shift(this->layer_adam_beta2 * this->layer_adam_velocity[index], this->layer_adam_beta_scale, this->layer_quant_params.weight_bits, false) + 
        requantize_shift( (128 - this->layer_adam_beta2) * avg_gradients[index] * avg_gradients[index], this->layer_adam_beta_scale, this->layer_quant_params.weight_bits, false);
        
        float m_hat = (this->layer_adam_momentum[index] / bias_correction1);
        float v_hat = (this->layer_adam_velocity[index] / bias_correction2);
        
        float weight_float = static_cast<float>(this->layer_weights[index]);
        weight_float -= pow(2,FUSION_QPARAM_WEIGHT_SCALE_SHIFT)*this->layer_adam_learning_rate * (m_hat / (sqrt(v_hat) + this->layer_adam_epsilon));
        //std::cout << pow(2,LAYER_10_QPARAM_WEIGHT_BITS-1)*this->layer_adam_learning_rate * (m_hat / (sqrt(v_hat) + this->layer_adam_epsilon)) <<std::endl;
        this->debug_float.push_back(weight_float);
        
       //s std::cout <<  std::to_string( pow(2,LAYER_10_QPARAM_WEIGHT_BITS)*this->layer_adam_learning_rate * (m_hat / (sqrt(v_hat) + this->layer_adam_epsilon))) << std::endl;
        this->layer_weights[index] = requantize_shift(static_cast<int32_t>(weight_float), 1.0f, LAYER_10_QPARAM_WEIGHT_BITS, false);
        
    }

    this->layer_adam_time_step++;   
}

void layer_q::stochastic_gradient_descent_optimize(const int32_t* layer_stochastic_gradient_descent_gradients_backprop, const uint32_t layer_stochastic_gradient_descent_size) {
    std::vector<int32_t> avg_gradients(layer_stochastic_gradient_descent_size, 0);

    for (uint32_t feature_index = 0; feature_index < layer_stochastic_gradient_descent_size; feature_index++) {
        int32_t sum = 0;
        for (uint32_t batch_index = 0; batch_index < this->layer_dim_size_in.batch; batch_index++) {
            sum += layer_stochastic_gradient_descent_gradients_backprop[feature_index * this->layer_dim_size_in.batch + batch_index];
        }
        avg_gradients[feature_index] = sum / this->layer_dim_size_in.batch;

    }


    for (uint32_t index = 0; index < layer_stochastic_gradient_descent_size; index++) {

        // Compute velocity
        this->layer_sgd_velocity[index] = requantize_shift(this->layer_sgd_momentum * this->layer_sgd_velocity[index], this->layer_sgd_momentum_scale, this->layer_quant_params.weight_bits, false)
                                    - requantize_shift(this->layer_sgd_learning_rate * avg_gradients[index], this->layer_sgd_momentum_scale, this->layer_quant_params.weight_bits, false);
        

        // Update gradient
#if DO_NESTEROV
        this->layer_weights[index] += requantize_shift(this->layer_sgd_momentum * this->layer_sgd_velocity[index], this->layer_sgd_momentum_scale, this->layer_quant_params.weight_bits, false)
        - requantize_shift(this->layer_sgd_learning_rate * avg_gradients[index], this->layer_sgd_momentum_scale, this->layer_quant_params.weight_bits, false); 
#else
        this->layer_weights[index] += this->layer_sgd_velocity[index]; 
#endif
    }

    this->layer_adam_time_step++;   
}
// getters
double layer_q::get_rescale_value() {
    return this->layer_rescale_value;
}
double layer_q::get_bw_rescale_value() {
    return this->layer_bw_rescale_value;
}
tensor_dim_sizes_t layer_q::get_input_size() {
    return this->layer_dim_size_in;
}
tensor_dim_sizes_t layer_q::get_output_size() {
    return this->layer_dim_size_out;
}
quant_param_t layer_q::get_qparams() {
    return this->layer_quant_params;
}
std::vector<int32_t> layer_q::get_weights() {
    return this->layer_weights;
}
std::vector<int32_t> layer_q::get_biases() {
    return this->layer_biases;
}
void layer_q::set_dense_bw_output_scale(float scale) {
    this->layer_bw_rescale_value = (this->layer_quant_params.scale_weight*(1.0f/255.0f))/scale;
}
LayerTypes layer_q::get_layer_type() {
    return this->layer_type;
}

std::vector<int32_t> layer_q::get_layer_bn_means() {
    return this->layer_bn_means;
}
std::vector<int32_t> layer_q::get_layer_bn_variances() {
    return this->layer_bn_variances;
}
