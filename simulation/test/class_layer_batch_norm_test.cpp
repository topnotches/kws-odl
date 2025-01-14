#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include "batch_norm_layer.hpp"

void test_batch_norm_sequential() {
    // Define input dimensions and parameters
    const uint16_t num_features = 2;
    const uint16_t num_batches = 4;

    // Initialize input features
    float input_features[num_features * num_batches] = {
        1.0, 2.5, 2.5, 4.0,  // Feature 1
        2.0, 4.0, 6.0, 8.0   // Feature 2
    };

    // Initialize gamma and beta
    float gamma[num_features] = {1.0, .50};
    float beta[num_features] = {0.0, 1.0};

    // Expected output features
    float expected_output_features[num_features * num_batches] = {
        -1.4142,0.0,0.0,1.4142,
        0.3292,0.7764,1.2236,1.6708
    };

    // Allocate memory for output features
    float output_features[num_features * num_batches];

    // Run the batch normalization layer
    batch_norm_sequential(input_features, output_features, gamma, beta, num_features, num_batches);

    // Validate the output features
    for (uint16_t feature = 0; feature < num_features; feature++) {
        for (uint16_t batch = 0; batch < num_batches; batch++) {
            uint16_t index = feature * num_batches + batch;
            // printf("Output: %f, Expected: %f\n", output_features[index], expected_output_features[index]);
            assert(fabs(output_features[index] - expected_output_features[index]) < 1e-4);
        }
    }

    printf("Batch normalization test passed!\n");
}

int main() {
    test_batch_norm_sequential();
    return 0;
}
