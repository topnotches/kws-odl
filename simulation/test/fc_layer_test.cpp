#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "fc_layer.hpp"
#include "misc_utils.hpp"


void test_fc_layer_sequential() {

    const uint8_t input_size = 200;
    const uint8_t output_size = 100;
    const uint8_t batch_size = 5;
    const float input_init_value = 2;
    const float weight_init_value = 2; 
    const float bias_init_value = 0; 

    float input_features[batch_size * input_size];
    float output_features[batch_size * output_size] = {};
    float expected_output_features[batch_size * output_size];

    float weights[input_size * output_size];
    float biases[input_size * output_size];

    init_flarr_to_num(weights, input_size * output_size, weight_init_value);
    init_flarr_to_num(biases, input_size * output_size, bias_init_value);
    init_flarr_to_num(input_features, input_size, input_init_value);


    init_flarr_to_num(expected_output_features, output_size, weight_init_value * input_init_value * input_size + input_size * bias_init_value);
    
    fc_layer(input_features, output_features, weights, biases, input_size, output_size, batch_size);


    // Validate the output features
    for (uint16_t batch_index = 0; batch_index < batch_size; batch_index++) {
        for (uint16_t out_index = 0; out_index < output_size; out_index++) {
            uint16_t index = batch_index * output_size + out_index;
            // printf("%d ", index);
            // printf("%f ", output_features[index]);
            // printf("%f \n", expected_output_features[index]);
            assert(fabs(output_features[index] - expected_output_features[index]) < 1e-6);
        
        }
    }

    printf("Fully-connected layer test passed!\n");
}

int main() {
    test_fc_layer_sequential();
    return 0;
}
