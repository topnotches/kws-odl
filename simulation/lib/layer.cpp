
#include "layer.hpp"
#include "batch_norm_layer.hpp"
#include "conv_layer.hpp"
#include "defs.hpp"
#include "dense_layer.hpp"
#include "dw_conv_layer.hpp"
#include "layer.hpp"
#include "misc_utils.hpp"
#include "relu_layer.hpp"


layer::layer(LayerTypes layer_type, tensor_dim_sizes_t layer_dim_size_in, float *weights = NULL, float *biases = NULL, conv_hypr_param_t layer_conv_hypr_params = {}) {

    this->layer_weights = {};
    this->layer_biases = {};
    this->layer_dim_size_in = {};  // {width, height, depth, batch_size}
    this->layer_dim_size_out = {}; // {width, height, depth, batch_size}¨
    this->layer_conv_hypr_params = {};
    this->layer_dense_conv_hypr_params = {};
    this->layer_outputs.resize(0);


    this->layer_dim_size_in = layer_dim_size_in; // {width, height, depth, batch_size}

    switch (layer_type) {
        case LayerTypes::conv: {
            this->layer_type = LayerTypes::conv; 

            this->layer_conv_hypr_params = layer_conv_hypr_params;

            this->layer_dim_size_out.width = (this->layer_dim_size_in.width - this->layer_conv_hypr_params.kernel_width + this->layer_conv_hypr_params.pad_left + this->layer_conv_hypr_params.pad_right) / this->layer_conv_hypr_params.kernel_stride + 1;
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

            uint32_t layer_conv_bias_count = layer_conv_weight_count;
            
            this->layer_weights.resize(layer_conv_weight_count);
            this->layer_weights.insert(this->layer_weights.end(), weights, weights + layer_conv_weight_count);

            this->layer_biases.resize(layer_conv_bias_count);
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

            uint32_t layer_conv_bias_count = layer_conv_weight_count;
            
            this->layer_weights.resize(layer_conv_weight_count);
            this->layer_weights.insert(this->layer_weights.end(), weights, weights + layer_conv_weight_count);

            this->layer_biases.resize(layer_conv_bias_count);
            this->layer_biases.insert(this->layer_biases.end(), biases, biases + layer_conv_bias_count);

            this->layer_outputs.resize(this->layer_dim_size_out.full);

            break;
        }
        case LayerTypes::dense: {
            this->layer_type = LayerTypes::dense;

            this->layer_dense_conv_hypr_params.size_in = this->layer_dim_size_in.full / this->layer_dim_size_in.batch;
            this->layer_dense_conv_hypr_params.size_out = layer_dense_conv_hypr_params.size_out;
            
            this->layer_dim_size_out.full = this->layer_dense_conv_hypr_params.size_out * this->layer_dim_size_in.batch;
            this->layer_dim_size_out.width = 0;
            this->layer_dim_size_out.height = 0;
            this->layer_dim_size_out.depth = 0;
            this->layer_dim_size_out.batch = this->layer_dim_size_in.batch;

            uint32_t layer_dense_weight_count = this->layer_dense_conv_hypr_params.size_in * this->layer_dense_conv_hypr_params.size_out;
            uint32_t layer_dense_bias_count = this->layer_dim_size_out.full / this->layer_dim_size_out.batch;

            this->layer_weights.resize(layer_dense_weight_count);
            this->layer_weights.insert(this->layer_weights.end(), weights, weights + layer_dense_weight_count);

            this->layer_biases.resize(layer_dense_bias_count);
            this->layer_biases.insert(this->layer_biases.end(), biases, biases + layer_dense_bias_count);

            this->layer_outputs.resize(this->layer_dim_size_out.full);

            break;
        }
        case LayerTypes::batchnorm: {

            this->layer_type = LayerTypes::batchnorm;
            this->layer_dim_size_out = this->layer_dim_size_in;
            
            uint32_t layer_bn_weight_count = this->layer_dim_size_in.full;
            uint32_t layer_bn_bias_count = this->layer_dim_size_in.full;

            this->layer_weights.resize(layer_bn_weight_count);
            this->layer_weights.insert(this->layer_weights.end(), weights, weights + layer_bn_weight_count);

            this->layer_biases.resize(layer_bn_bias_count);
            this->layer_biases.insert(this->layer_biases.end(), biases, biases + layer_bn_bias_count);

            this->layer_outputs.resize(this->layer_dim_size_out.full);

            break;
          }  
        case LayerTypes::relu: {
            this->layer_type = LayerTypes::relu;

            this->layer_dim_size_out = layer_dim_size_in;

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


void layer::forward(float *layer_input) {
    switch (layer_type) {
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
                        this->layer_dense_conv_hypr_params.size_in, this->layer_dense_conv_hypr_params.size_in, this->layer_dim_size_out.batch);
            break;
        }
        case LayerTypes::batchnorm: {

            batch_norm_sequential(layer_input, this->layer_outputs.data(), this->layer_weights.data(), this->layer_biases.data(),
                                    this->layer_dim_size_in.full / this->layer_dim_size_in.batch, this->layer_dim_size_in.batch);

            break;
          }  
        case LayerTypes::relu: {
            relu_layer(layer_input, this->layer_outputs.data(), this->layer_dim_size_in.width, this->layer_dim_size_in.height, this->layer_dim_size_in.depth);
            break;
        }
        default: {
            KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("Trying forward() on an unsupported layer...");
            break;
        }  
    }
}