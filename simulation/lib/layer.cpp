
#include "layer.hpp"
#include "defs.hpp"
#include <iostream>
#include "misc_utils.hpp"

#include "batch_norm_layer.hpp"
#include "conv_layer.hpp"
#include "dense_layer.hpp"
#include "dw_conv_layer.hpp"
#include "relu_layer.hpp"
#include "avgpool2d_layer.hpp"
#include "softmax_layer.hpp"
#include "cross_entropy_loss_layer.hpp"


layer::layer(LayerTypes         layer_type, 
          tensor_dim_sizes_t    layer_dim_size_in, 
          float                 *weights, 
          float                 *biases, 
          conv_hypr_param_t     layer_conv_hypr_params,
          dense_hypr_param_t    layer_dense_hypr_params,
          float                 *layer_bn_means,
          float                 *layer_bn_variances) {

    this->layer_weights = {};
    this->layer_biases = {};
    this->layer_dim_size_in = {};  // {width, height, depth, batch_size}
    this->layer_dim_size_out = {}; // {width, height, depth, batch_size}
    this->layer_conv_hypr_params = {};
    this->layer_dense_hypr_params = {};
    this->layer_outputs.resize(0);


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

            break;
        }
        case LayerTypes::dense: {
            this->layer_type = LayerTypes::dense;

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

            break;
          }  
        case LayerTypes::relu: {
            this->layer_type = LayerTypes::relu;

            this->layer_dim_size_out = this->layer_dim_size_in;

            this->layer_outputs.resize(this->layer_dim_size_out.full);

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

            break;
        }
        case LayerTypes::softmax: {
            this->layer_type = LayerTypes::softmax;
            
            this->layer_dim_size_out = this->layer_dim_size_in;
            
            this->layer_outputs.resize(this->layer_dim_size_out.full);

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

            break;
        }
        default: {
            KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("Trying to initialize an unsupported layer...");
            break;
        }  
    }
    
}

layer::~layer() {

}


void layer::forward(float *layer_input, float *labels_input) {
    switch (this->layer_type) {
        case LayerTypes::conv: {
            conv_layer_sequential(layer_input, this->layer_outputs.data(), this->layer_weights.data(), this->layer_biases.data(),
                            this->layer_dim_size_in.width, this->layer_dim_size_in.height, this->layer_dim_size_in.depth,
                            this->layer_conv_hypr_params.kernel_stride, this->layer_conv_hypr_params.kernel_width, this->layer_conv_hypr_params.kernel_height,
                            this->layer_conv_hypr_params.kernel_count, this->layer_dim_size_out.batch,
                            this->layer_conv_hypr_params.pad_top, this->layer_conv_hypr_params.pad_bottom,
                            this->layer_conv_hypr_params.pad_left, this->layer_conv_hypr_params.pad_right);
            break;
        }
        case LayerTypes::dw_conv: {
            dw_conv_layer_sequential(layer_input, this->layer_outputs.data(), this->layer_weights.data(), this->layer_biases.data(),
                                        this->layer_dim_size_in.width, this->layer_dim_size_in.height, this->layer_dim_size_in.depth,
                                        this->layer_conv_hypr_params.kernel_stride, this->layer_conv_hypr_params.kernel_width,
                                        this->layer_conv_hypr_params.kernel_height, this->layer_dim_size_in.batch,
                                        this->layer_conv_hypr_params.pad_top, this->layer_conv_hypr_params.pad_bottom,
                                        this->layer_conv_hypr_params.pad_left, this->layer_conv_hypr_params.pad_right);

            break;
        }
        case LayerTypes::dense: {

            dense_layer(layer_input, this->layer_outputs.data(), this->layer_weights.data(), this->layer_biases.data(),
                        this->layer_dense_hypr_params.size_in, this->layer_dense_hypr_params.size_out, this->layer_dim_size_out.batch);

            break;
        }
        case LayerTypes::batchnorm: {

            batch_norm_sequential(layer_input, this->layer_outputs.data(), this->layer_weights.data(), this->layer_biases.data(),
                                    this->layer_bn_means.data(), this->layer_bn_variances.data(),
                                    this->layer_dim_size_in.width * this->layer_dim_size_in.height, this->layer_dim_size_in.depth, this->layer_dim_size_in.batch);

            break;
          }  
        case LayerTypes::relu: {

            relu_layer(layer_input, this->layer_outputs.data(), this->layer_dim_size_in.width, this->layer_dim_size_in.height, this->layer_dim_size_in.depth*this->layer_dim_size_in.batch);
            break;
        }
        case LayerTypes::avgpool2d: {
            avgpool2d_layer_sequential(layer_input, this->layer_outputs.data(),
                                        this->layer_dim_size_in.width, this->layer_dim_size_in.height, this->layer_dim_size_in.depth,
                                        this->layer_conv_hypr_params.kernel_stride, this->layer_conv_hypr_params.kernel_width,
                                        this->layer_conv_hypr_params.kernel_height, this->layer_dim_size_in.batch,
                                        this->layer_conv_hypr_params.pad_top, this->layer_conv_hypr_params.pad_bottom,
                                        this->layer_conv_hypr_params.pad_left, this->layer_conv_hypr_params.pad_right);
            break;
        }
        case LayerTypes::softmax: {
            softmax_layer_sequential(layer_input, this->layer_outputs.data(), this->layer_dim_size_in.batch, this->layer_dim_size_in.width);
            break;
        }
        case LayerTypes::cross_entropy_loss: {
            cross_entropy_loss_sequential(labels_input, layer_input, this->layer_outputs.data(),  this->layer_dim_size_out.batch, this->layer_dim_size_in.width);
            break;
        }
        default: {
            KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("Trying forward() on an unsupported layer...");
            break;
        }  
    }
}

// getters
tensor_dim_sizes_t layer::get_input_size() {
    return this->layer_dim_size_in;
}
tensor_dim_sizes_t layer::get_output_size() {
    return this->layer_dim_size_out;
}
std::vector<float> layer::get_weights() {
    return this->layer_weights;
}
std::vector<float> layer::get_biases() {
    return this->layer_biases;
}
LayerTypes layer::get_layer_type() {
    return this->layer_type;
}
std::vector<float> layer::get_layer_bn_means() {
    return this->layer_bn_means;
}
std::vector<float> layer::get_layer_bn_variances() {
    return this->layer_bn_variances;
}
