#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include "batch_norm_layer.hpp"
#include "misc_utils.hpp"
#include "layer.hpp"

void test_batch_norm_float() {
    // Define input dimensions and parameters
    const uint16_t num_features = 2;
    const uint16_t num_batches = 4;

    // Initialize input features
    float input_features[num_features * num_batches] = {
        1.0, 2.5, 2.5, 4.0,  // Feature 1
        2.0, 4.0, 6.0, 8.0   // Feature 2
    };

    // Define input size for batchnorm layer constructor param

    tensor_dim_sizes_t layer_dim_size_in;

    layer_dim_size_in.width = 0;
    layer_dim_size_in.height = 0;
    layer_dim_size_in.depth = 0;
    layer_dim_size_in.batch = num_batches;
    layer_dim_size_in.full = num_features *
                            layer_dim_size_in.batch;
    // Initialize gamma and beta
    float gamma[num_features] = {1.0, .50};
    float beta[num_features] = {0.0, 1.0};

    // Expected output features
    float expected_output_features[num_features * num_batches] = {
        -1.4142,0.0,0.0,1.4142,
        0.3292,0.7764,1.2236,1.6708
    };

    layer my_layer(LayerTypes::batchnorm, layer_dim_size_in, gamma, beta);

    // Run the ReLU layer
    my_layer.forward(input_features);

    // Validate the output features
    for (uint16_t feature = 0; feature < num_features; feature++) {
        for (uint16_t batch = 0; batch < num_batches; batch++) {
            uint16_t index = feature * num_batches + batch;
            // printf("Output: %f, Expected: %f\n", my_layer.layer_outputs[index], expected_output_features[index]);
            assert(fabs(my_layer.layer_outputs[index] - expected_output_features[index]) < 1e-4);
        }
    }

    printf("Batch normalization test passed!\n");
}

int main() {
    test_batch_norm_float();
    return 0;
}
