#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "conv_layer.hpp"

void test_conv_layer_sequential() {
    // Define input dimensions and parameters
    const uint16_t input_width = 4;
    const uint16_t input_height = 4;
    const uint16_t input_depth = 4;
    const uint8_t stride = 1;
    const uint8_t kernel_width = 3;
    const uint8_t kernel_height = 3;
    const uint8_t output_feature_count = 4;

    // Calculate output dimensions
    const uint16_t output_width = (input_width - kernel_width) / stride + 1;
    const uint16_t output_height = (input_height - kernel_height) / stride + 1;

    // Initialize input features
    float input_features[input_width * input_height * input_depth] = {
        0, 1, 2, 3,
        1, 0, 1, 2,
        2, 1, 0, 1,
        3, 2, 1, 0,

        0, 1, 2, 3,
        1, 0, 1, 2,
        2, 1, 0, 1,
        3, 2, 1, 0,

        3, 2, 1, 0,
        2, 1, 0, 1,
        1, 0, 1, 2,
        0, 1, 2, 3,
            
        3, 2, 1, 0,
        2, 1, 0, 1,
        1, 0, 1, 2,
        0, 1, 2, 3
    };

    // Initialize convolution kernel
    float conv_kernels[kernel_width * kernel_height * input_depth * output_feature_count] = {
        // kernel 1
        0,-1,-2,
        1, 0,-1,
        2, 1, 0,

        0,-1,-2,
        1, 0,-1,
        2, 1, 0,

        0,-1,-2,
        1, 0,-1,
        2, 1, 0,
        
        0,-1,-2,
        1, 0,-1,
        2, 1, 0,

        // kernel 2
         0, 1, 2,
        -1, 0, 1,
        -2,-1, 0,

         0, 1, 2,
        -1, 0, 1,
        -2,-1, 0,

         0, 1, 2,
        -1, 0, 1,
        -2,-1, 0,

         0, 1, 2,
        -1, 0, 1,
        -2,-1, 0,

        // kernel 3
        2, 1, 0,
        1, 0,-1,
        0,-1,-2,

        2, 1, 0,
        1, 0,-1,
        0,-1,-2,

        2, 1, 0,
        1, 0,-1,
        0,-1,-2,

        2, 1, 0,
        1, 0,-1,
        0,-1,-2,

        // kernel 4
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,

        1, 1, 1,
        1, 1, 1,
        1, 1, 1,

        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
        
    };

    // Expected output features
    float expected_output_features[output_width * output_height * output_feature_count] = {
        0.000000, -16.000000, 
        16.000000, 0.000000, 
        0.000000, 16.000000, 
        -16.000000, 0.000000, 
        16.000000, 0.000000, 
        0.000000, -16.000000, 
        38.000000, 38.000000, 
        38.000000, 38.000000
    };

    // Allocate memory for output features
    float output_features[output_width * output_height * output_feature_count];

    // Run the convolution layer
    conv_layer_sequential(input_features, output_features, conv_kernels,
                          input_width, input_height, input_depth,
                          stride, kernel_width, kernel_height, output_feature_count);

    // Validate the output features
    for (uint16_t slopper = 0; slopper < output_feature_count; slopper++) {
        for (uint16_t row = 0; row < output_height; row++) {
            for (uint16_t col = 0; col < output_width; col++) {
                uint16_t index = slopper * output_width * output_height + row * output_width + col;
                // printf("%f ", output_features[index]);
                assert(fabs(output_features[index] - expected_output_features[index]) < 1e-6);
            }
            // printf("\n");
        }
    }

    printf("Convolution test passed!\n");
}

int main() {
    test_conv_layer_sequential();
    return 0;
}
