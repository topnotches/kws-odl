#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "avgpool2d_layer.hpp"
#include "misc_utils.hpp"
#include "layer.hpp"

void test_avgpool2d_layer_sequential() {
    // Define input dimensions and parameters
    const uint16_t input_width = 4;
    const uint16_t input_height = 4;
    const uint16_t input_depth = 3;
    const uint8_t stride = 2;
    const uint8_t kernel_width = 3;
    const uint8_t kernel_height = 3;
    const uint8_t batch_size = 2;
    // Padding
    const uint8_t pad_top = 1;
    const uint8_t pad_bottom = 0;
    const uint8_t pad_left = 1;
    const uint8_t pad_right = 0;
    // Calculate output dimensions
    const uint16_t output_width = (input_width - kernel_width + pad_left + pad_right) / stride + 1;
    const uint16_t output_height = (input_height - kernel_height + pad_top + pad_bottom) / stride + 1;
    // printf("\nheight: %d \n", output_width);
    // printf("width: %d \n", output_height);

    // Define input size for avgpool2d-conv layer constructor param

    tensor_dim_sizes_t layer_dim_size_in;

    layer_dim_size_in.width = input_width;
    layer_dim_size_in.height = input_height;
    layer_dim_size_in.depth = input_depth;
    layer_dim_size_in.batch = batch_size;
    layer_dim_size_in.full = layer_dim_size_in.width *
                            layer_dim_size_in.height *
                            layer_dim_size_in.depth *
                            layer_dim_size_in.batch;
                            
    // Define conv kernel hyper parameters;

    conv_hypr_param_t conv_params;

    conv_params.pad_top = pad_top; 
    conv_params.pad_bottom = pad_bottom; 
    conv_params.pad_left = pad_left; 
    conv_params.pad_right = pad_right; 
    conv_params.kernel_stride = stride;
    conv_params.kernel_width = kernel_width;
    conv_params.kernel_height = kernel_height;

    float input_features[batch_size * input_width * input_height * input_depth] = {
        // avg pool kernels
        1,1,1,1,
        2,2,2,2,
        2,2,2,2,
        1,1,1,1,
        
        2,2,2,2,
        2,2,2,2,
        2,2,2,2,
        2,2,2,2,

        3,3,3,3,
        2,2,2,2,
        2,2,2,2,
        1,1,1,1,
        // avg pool kernels
        1,1,1,1,
        2,2,2,2,
        2,2,2,2,
        1,1,1,1,
        
        2,2,2,2,
        2,2,2,2,
        2,2,2,2,
        2,2,2,2,

        3,3,3,3,
        2,2,2,2,
        2,2,2,2,
        1,1,1,1,
    };

    // Expected output features
    double expected_output_features[batch_size * output_width * output_height * input_depth] = {
        // Batch 0
        
        1.5, 1.5,
        1.6666666, 1.6666666,
        
        2, 2,
        2, 2,
        
        2.5, 2.5,
        1.66, 1.66,

        // Batch 1

        1.5, 1.5,
        1.6666666, 1.6666666,
        
        2, 2,
        2, 2,
        
        2.5, 2.5,
        5/3, 5/3

    };

    layer my_layer(LayerTypes::avgpool2d, layer_dim_size_in, {}, {}, conv_params);

    // Run the depthwise convolution layer
    my_layer.forward(input_features);

    // Validate the output features
    for (uint16_t batch = 0; batch < batch_size; batch++) {
        for (uint16_t map = 0; map < input_depth; map++) {
            for (uint16_t row = 0; row < output_height; row++) {
                for (uint16_t col = 0; col < output_width; col++) {
                    uint16_t index = map * output_width * output_height + row * output_width + col;
                    // printf("%f ", my_layer.layer_outputs[index]);


                    // printf("%f ", my_layer.layer_outputs[index]);
                    // printf("%f EXP \n", expected_output_features[index]);
                    assert(fabs(my_layer.layer_outputs[index] - expected_output_features[index]) < 1e-2); // float on intel ????
                }
            }
        }
    }
    printf("Layer Class: Average pooling stride test passed!\n");
}

int main() {
    test_avgpool2d_layer_sequential();
    return 0;
}
