#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "relu_layer.hpp"

void test_fc_layer_sequential() {
    // Define input dimensions and parameters
    const uint16_t input_width = 4;
    const uint16_t input_height = 4;
    const uint8_t stride = 1;
    const uint8_t kernel_width = 3;
    const uint8_t kernel_height = 3;
    const uint8_t output_feature_count = 4;
    const uint8_t batch_size = 2;
    const uint8_t relu_depth = batch_size * output_feature_count;

    // Calculate output dimensions
    const uint16_t output_width = (input_width - kernel_width) / stride + 1;
    const uint16_t output_height = (input_height - kernel_height) / stride + 1;

    // Initialize input features
    float input_features[batch_size * output_width * output_height * output_feature_count] = {
        0.000000, -16.000000, 
        16.000000, 0.000000, 
        0.000000, 16.000000, 
        -16.000000, 0.000000,
        
        16.000000, 0.000000, 
        0.000000, -16.000000, 
        38.000000, 38.000000, 
        38.000000, 38.000000,
        
        0.000000, -16.000000, 
        16.000000, 0.000000, 
        0.000000, 16.000000, 
        -16.000000, 0.000000, 
        
        16.000000, 0.000000, 
        0.000000, -16.000000, 
        38.000000, 38.000000, 
        38.000000, 38.000000
    };

    // Expected output features
    float expected_output_features[batch_size * output_width * output_height * output_feature_count] = {
        0.000000, 0.0000000, 
        16.000000, 0.000000, 
        0.000000, 16.000000, 
        0.0000000, 0.000000, 
        16.000000, 0.000000, 
        0.000000, 0.0000000, 
        38.000000, 38.000000, 
        38.000000, 38.000000,
        0.000000, 0.0000000, 
        16.000000, 0.000000, 
        0.000000, 16.000000, 
        0.0000000, 0.000000, 
        16.000000, 0.000000, 
        0.000000, 0.0000000, 
        38.000000, 38.000000, 
        38.000000, 38.000000
    };

    // Allocate memory for output features
    float output_features[batch_size * output_width * output_height * output_feature_count];

    // Run the convolution layer
    relu_layer(input_features, output_features, output_width, output_height, relu_depth);

    // Validate the output features
    for (uint16_t slopper = 0; slopper < relu_depth; slopper++) {
        for (uint16_t row = 0; row < output_height; row++) {
            for (uint16_t col = 0; col < output_width; col++) {
                uint16_t index = slopper * output_width * output_height + row * output_width + col;
                // printf("%f ", output_features[index]);
                // printf("%f \n", expected_output_features[index]);
                assert(fabs(output_features[index] - expected_output_features[index]) < 1e-6);
            }
            // printf("\n");
        }
    }

    printf("Relu layer test passed!\n");
}

int main() {
    test_fc_layer_sequential();
    return 0;
}
