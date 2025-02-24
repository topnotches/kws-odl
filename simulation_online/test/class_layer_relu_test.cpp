#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "relu_layer.hpp"
#include "misc_utils.hpp"
#include "layer.hpp"

void test_dense_layer_float() {
    // Define input dimensions and parameters
    const uint16_t input_width = 2;
    const uint16_t input_height = 4;
    const uint8_t input_depth = 2;
    const uint8_t batch_size = 2;
    const uint8_t relu_depth = batch_size * input_depth;

    // Define input size for relu layer constructor param

    tensor_dim_sizes_t layer_dim_size_in;

    layer_dim_size_in.width = input_width;
    layer_dim_size_in.height = input_height;
    layer_dim_size_in.depth = input_depth;
    layer_dim_size_in.batch = batch_size;
    layer_dim_size_in.full = layer_dim_size_in.width *
                            layer_dim_size_in.height *
                            layer_dim_size_in.depth *
                            layer_dim_size_in.batch;

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


    layer my_layer(LayerTypes::relu, layer_dim_size_in);

    // Run the ReLU layer
    my_layer.forward(input_features);

    // Validate the output features
    for (uint16_t slopper = 0; slopper < relu_depth; slopper++) {
        for (uint16_t row = 0; row < input_height; row++) {
            for (uint16_t col = 0; col < input_width; col++) {
                uint16_t index = slopper * input_width * input_height + row * input_width + col;
                // printf("%f ", my_layer.layer_outputs[index]);
                // printf("%f \n", expected_output_features[index]);
                assert(fabs(my_layer.layer_outputs[index] - expected_output_features[index]) < 1e-6);
            }
            // printf("\n");
        }
    }

    printf("Layer Class: Relu layer test passed!\n");
}

int main() {
    test_dense_layer_float();
    return 0;
}
