#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "relu_layer.hpp"

void test_dense_layer_sequential() {
    // Define input dimensions and parameters
    const uint16_t input_width = 2;
    const uint16_t input_height = 4;
    const uint8_t input_depth = 2;
    const uint8_t batch_size = 2;
    const uint8_t relu_depth = batch_size * input_depth;

    // // Calculate output dimensions
    // printf("\n%d\n", batch_size * input_width * input_height * input_depth);
    // printf("\n%d\n", batch_size * input_width * input_height * input_depth);
    // printf("\n%d\n", batch_size * input_width * input_height * input_depth);
    // printf("\n%d\n", batch_size * input_width * input_height * input_depth);

    // Initialize input features
    float input_features[batch_size * input_width * input_height * input_depth] = {
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
    float expected_output_features[batch_size * input_width * input_height * input_depth] = {
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
    float output_features[batch_size * input_width * input_height * input_depth];

    // Run the convolution layer
    relu_layer(input_features, output_features, input_width, input_height, relu_depth);

    // Validate the output features
    for (uint16_t slopper = 0; slopper < relu_depth; slopper++) {
        for (uint16_t row = 0; row < input_height; row++) {
            for (uint16_t col = 0; col < input_width; col++) {
                uint16_t index = slopper * input_width * input_height + row * input_width + col;
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
    test_dense_layer_sequential();
    return 0;
}
