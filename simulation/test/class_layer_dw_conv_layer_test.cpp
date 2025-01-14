#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "dw_conv_layer.hpp"
#include "misc_utils.hpp"

void test_conv_layer_sequential() {
    // Define input dimensions and parameters
    const uint16_t input_width = 4;
    const uint16_t input_height = 4;
    const uint16_t input_depth = 3;
    const uint8_t stride = 1;
    const uint8_t kernel_width = 3;
    const uint8_t kernel_height = 3;
    const uint8_t batch_size = 2;
    const uint8_t bias_init = 123;
    // Padding
    const uint8_t pad_top = 0;
    const uint8_t pad_bottom = 0;
    const uint8_t pad_left = 0;
    const uint8_t pad_right = 0;
    // Calculate output dimensions
    const uint16_t output_width =   (input_width - kernel_width + pad_left + pad_right) / stride + 1;
    const uint16_t output_height =  (input_height - kernel_height + pad_top + pad_bottom) / stride + 1;
    // printf("\nheight: %d \n", output_width);
    // printf("width: %d \n", output_height);

    // Initialize input features
    float input_features[batch_size * input_width * input_height * input_depth] = {

        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,

        2, 2, 2, 2,
        2, 2, 2, 2,
        2, 2, 2, 2,
        2, 2, 2, 2,

        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
        3, 3, 3, 3,
        
        // Batch 1

        4, 4, 4, 4,
        4, 4, 4, 4,
        4, 4, 4, 4,
        4, 4, 4, 4,

        5, 5, 5, 5,
        5, 5, 5, 5,
        5, 5, 5, 5,
        5, 5, 5, 5,

        6, 6, 6, 6,
        6, 6, 6, 6,
        6, 6, 6, 6,
        6, 6, 6, 6
    };

    // Initialize convolution kernel
    float conv_biases[kernel_width * kernel_height * input_depth];
    init_flarr_to_num(conv_biases, kernel_width * kernel_height * input_depth, bias_init);
    
    float conv_weights[kernel_width * kernel_height * input_depth] = {

        // DW kernels

        1,1,1,
        1,1,1,
        1,1,1,

        1,1,1,
        1,1,1,
        1,1,1,

        1,1,1,
        1,1,1,
        1,1,1
        
    };

    // Expected output features
    float expected_output_features[batch_size * output_width * output_height * input_depth] = {

        // Batch 0

        9,9,
        9,9,

        9*2,9*2,
        9*2,9*2,

        9*3,9*3,
        9*3,9*3,

        // Batch 1

        10*4,10*4,
        10*4,10*4,

        9*5,9*5,
        9*5,9*5,

        9*6,9*6,
        9*6,9*6,
    };

    // Allocate memory for output features
    float output_features[batch_size * output_width * output_height * input_depth];

    // Run the convolution layer
    dw_conv_layer_sequential(input_features, output_features, conv_weights,conv_biases,
                          input_width, input_height, input_depth,
                          stride, kernel_width, kernel_height, batch_size,
                          pad_top, pad_bottom, pad_left, pad_right);

    // Validate the output features
    for (uint16_t batch = 0; batch < batch_size; batch++) {

        for (uint16_t map = 0; map < input_depth; map++) {
            for (uint16_t row = 0; row < output_height; row++) {
                for (uint16_t col = 0; col < output_width; col++) {
                    uint16_t index = map * output_width * output_height + row * output_width + col;
                    // printf("%f ", output_features[index]);
                    // printf("%f \n", expected_output_features[index]+kernel_width * kernel_height * bias_init);
                    assert(fabs(output_features[index] - (expected_output_features[index]+kernel_width * kernel_height * bias_init)) < 1e-6);
                }
                // printf("\n");
            }
        }
    }
    printf("Depthwise convolution test passed!\n");
}

int main() {
    test_conv_layer_sequential();
    return 0;
}
