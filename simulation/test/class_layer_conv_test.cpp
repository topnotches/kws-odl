#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "conv_layer.hpp"
#include "misc_utils.hpp"

void test_conv_layer_sequential() {
    // Define input dimensions and parameters
    const uint16_t input_width = 4;
    const uint16_t input_height = 4;
    const uint16_t input_depth = 4;
    const uint8_t stride = 1;
    const uint8_t kernel_width = 3;
    const uint8_t kernel_height = 3;
    const uint8_t output_feature_map_count = 4;
    const uint8_t batch_size = 2;
    const uint8_t bias_init = 23;

    const uint8_t pad_top = 0;
    const uint8_t pad_bottom = 0;
    const uint8_t pad_left = 0;
    const uint8_t pad_right = 0;
    // Calculate output dimensions
    const uint16_t output_width =   (input_width - kernel_width + pad_left + pad_right) / stride + 1;
    const uint16_t output_height =  (input_height - kernel_height + pad_top + pad_bottom) / stride + 1;

    // Initialize input features
    float input_features[batch_size * input_width * input_height * input_depth] = {
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
        0, 1, 2, 3,
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
        0, 1, 2, 3,
    };

    // Initialize convolution kernel
    float conv_biases[kernel_width * kernel_height * input_depth * output_feature_map_count];
    init_flarr_to_num(conv_biases, kernel_width * kernel_height * input_depth * output_feature_map_count, bias_init);
    
    float conv_weights[kernel_width * kernel_height * input_depth * output_feature_map_count] = {
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
        1, 1, 1
        
    };

    // Expected output features
    float expected_output_features[batch_size * output_width * output_height * output_feature_map_count] = {
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

    // Allocate memory for output features
    float output_features[batch_size * output_width * output_height * output_feature_map_count];

    // Run the convolution layer
    conv_layer_sequential(input_features, output_features, conv_weights,conv_biases,
                          input_width, input_height, input_depth,
                          stride, kernel_width, kernel_height, output_feature_map_count, batch_size,
                          pad_top, pad_bottom, pad_left, pad_right);

    // Validate the output features
    for (uint16_t batch = 0; batch < batch_size; batch++) {
        //printf("\n");
        for (uint16_t map = 0; map < output_feature_map_count; map++) {
            for (uint16_t row = 0; row < output_height; row++) {
                for (uint16_t col = 0; col < output_width; col++) {
                    uint16_t index = batch * output_feature_map_count * output_height * output_width + map * output_width * output_height + row * output_width + col;
                    // printf("%f ", output_features[index]);
                    // printf("%f \n", (expected_output_features[index]+kernel_width * kernel_height * input_depth * bias_init);
                    assert(fabs(output_features[index] - (expected_output_features[index]+kernel_width * kernel_height * input_depth * bias_init)) < 1e-6);
                }
            }
        }
    }

    printf("Convolution test passed!\n");
}

int main() {
    test_conv_layer_sequential();
    return 0;
}
