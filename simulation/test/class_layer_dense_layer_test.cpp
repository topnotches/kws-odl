#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "dense_layer.hpp"
#include "misc_utils.hpp"
#include "layer.hpp"


void test_dense_layer_sequential() {

    const uint8_t input_size = 200;
    const uint8_t output_size = 100;
    const uint8_t batch_size = 5;
    const float input_init_value = 2;
    const float weight_init_value = 2; 
    const float bias_init_value = 123; 

    float input_features[batch_size * input_size];
    float expected_output_features[batch_size * output_size];

    float weights[input_size * output_size];
    float biases[input_size * output_size];

    // Define input size for dense layer constructor param

    tensor_dim_sizes_t layer_dim_size_in;

    layer_dim_size_in.batch = batch_size;
    layer_dim_size_in.full = input_size*batch_size;

    // define conv kernel hyper parameters;

    dense_param_t dense_params;

    dense_params.size_out = output_size;

    init_flarr_to_num(weights, input_size * output_size, weight_init_value);
    init_flarr_to_num(biases, input_size * output_size, bias_init_value);
    init_flarr_to_num(input_features, input_size*batch_size, input_init_value);


    init_flarr_to_num(expected_output_features, output_size*batch_size, weight_init_value * input_init_value * input_size + bias_init_value);
    
    layer my_layer(LayerTypes::dense, layer_dim_size_in, weights, biases, {}, dense_params);

    // Run the dense layer
    my_layer.forward(input_features);

    // Validate the output features
    for (uint16_t batch_index = 0; batch_index < batch_size; batch_index++) {
        for (uint16_t out_index = 0; out_index < output_size; out_index++) {
            uint16_t index = batch_index * output_size + out_index;
            // printf("%d ", index);
            // printf("%f ", my_layer.layer_outputs[index]);
            // printf("%f \n", expected_output_features[index]);
            assert(fabs(my_layer.layer_outputs[index] - expected_output_features[index]) < 1e-6);
        
        }
    }

    printf("Layer Class: Dense layer test passed!\n");

}

int main() {
    test_dense_layer_sequential();
    return 0;
}
