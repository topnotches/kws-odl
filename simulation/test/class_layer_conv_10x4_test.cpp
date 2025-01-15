#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "conv_layer.hpp"
#include "misc_utils.hpp"
#include "defs.hpp"
#include "layer.hpp"

void test_conv_layer_sequential() {
    // Define input dimensions and parameters
    const uint16_t input_width = 12;
    const uint16_t input_height = 4;
    const uint16_t input_depth = 2;
    const uint8_t stride = 2;
    const uint8_t kernel_width = 10;
    const uint8_t kernel_height = 4;
    const uint8_t output_feature_map_count = 4;
    const uint8_t batch_size = 2;
    const uint8_t bias_init = 0;

    const uint8_t pad_top = 0;
    const uint8_t pad_bottom = 0;
    const uint8_t pad_left = 0;
    const uint8_t pad_right = 0;
    // Calculate output dimensions
    const uint16_t output_width =   (input_width - kernel_width + pad_left + pad_right) / stride + 1;
    const uint16_t output_height =  (input_height - kernel_height + pad_top + pad_bottom) / stride + 1;
    printf("%f, output_width \n", (float)(input_width - kernel_width + pad_left + pad_right) / stride + 1);
    printf("%f, output_height \n", (float)(input_height - kernel_height + pad_top + pad_bottom) / stride + 1);
    // Define input size for conv layer constructor param

    tensor_dim_sizes_t layer_dim_size_in;

    layer_dim_size_in.width = input_width;
    layer_dim_size_in.height = input_height;
    layer_dim_size_in.depth = input_depth;
    layer_dim_size_in.batch = batch_size;
    layer_dim_size_in.full = layer_dim_size_in.width *
                            layer_dim_size_in.height *
                            layer_dim_size_in.depth *
                            layer_dim_size_in.batch;

    // define conv kernel hyper parameters

    conv_hypr_param_t conv_params;

    conv_params.pad_top = pad_top; 
    conv_params.pad_bottom = pad_bottom; 
    conv_params.pad_left = pad_left; 
    conv_params.pad_right = pad_right; 
    conv_params.kernel_stride = stride;
    conv_params.kernel_width = kernel_width;
    conv_params.kernel_height = kernel_height;
    conv_params.kernel_count = output_feature_map_count;

    // Initialize input features
    float input_features[batch_size * input_width * input_height * input_depth] = {
        
        // Batch 0

        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 

        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
        
        // Batch 1

        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,

        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,

    };

    // Initialize convolution kernel
    float conv_biases[kernel_width * kernel_height * input_depth * output_feature_map_count];
    init_flarr_to_num(conv_biases, kernel_width * kernel_height * input_depth * output_feature_map_count, bias_init);
    
    float conv_weights[kernel_width * kernel_height * input_depth * output_feature_map_count] = {

        // kernel 0

        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,

        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,

        // kernel 1

        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,

        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,

        // kernel 2

        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,

        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,

        // kernel 3

        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,

        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        
    };

    // Expected output features
    float expected_output_features[batch_size * output_width * output_height * output_feature_map_count] = {
        
        // batch 0
        
        120, 120,
        120, 120,
        
        120, 120,
        120, 120,
        
        
        // batch 0

        280, 280,
        280, 280,

        280, 280,
        280, 280,


    };

    layer my_layer(LayerTypes::conv, layer_dim_size_in, conv_weights, conv_biases, conv_params = conv_params);

    // Run the dw convolution layer
    my_layer.forward(input_features);


    // Validate the output features
    for (uint16_t batch = 0; batch < batch_size; batch++) {
        //printf("\n");
        for (uint16_t map = 0; map < output_feature_map_count; map++) {
            for (uint16_t row = 0; row < output_height; row++) {
                for (uint16_t col = 0; col < output_width; col++) {
                    uint16_t index = batch * output_feature_map_count * output_height * output_width + map * output_width * output_height + row * output_width + col;
                    // printf("index: %d, ", index);
                    // printf("%f ", my_layer.layer_outputs[index]);
                    // printf("%f \n", (expected_output_features[index]+kernel_width * kernel_height * input_depth * bias_init));
                    assert(fabs(my_layer.layer_outputs[index] - (expected_output_features[index]+kernel_width * kernel_height * input_depth * bias_init)) < 1e-6);
                }
            }
        }
    }

    printf("Layer Class: Convolution padding test passed!\n");
}

int main() {
    test_conv_layer_sequential();
    return 0;
}
