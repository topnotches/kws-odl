#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "batch_norm_layer.hpp"

void batch_norm_sequential(float *bn_input_features, float *bn_output_features, float *bn_gamma,
                        float *bn_beta, const uint16_t bn_num_features, const uint16_t bn_num_batches) {

    const float epsilon = 1e-5; // Small constant for numerical stability
    float bn_means[bn_num_features];
    float bn_variances[bn_num_features];

    // Initialize means and variances to 0
    memset(bn_means, 0, sizeof(float) * bn_num_features);
    memset(bn_variances, 0, sizeof(float) * bn_num_features);

    // Calculate feature means
    #pragma omp parallel for reduction(+:bn_means[:bn_num_features]) schedule(dynamic)
    for (uint16_t index_feature = 0; index_feature < bn_num_features; index_feature++) {
        for (uint16_t index_batch = 0; index_batch < bn_num_batches; index_batch++) {
            uint16_t index = index_feature * bn_num_batches + index_batch;
            bn_means[index_feature] += bn_input_features[index];
        }
    }
    for (uint16_t index_feature = 0; index_feature < bn_num_features; index_feature++) {
        bn_means[index_feature] /= bn_num_batches;
    }

    // Calculate feature variances
    #pragma omp parallel for reduction(+:bn_variances[:bn_num_features]) schedule(dynamic)
    for (uint16_t index_feature = 0; index_feature < bn_num_features; index_feature++) {
        for (uint16_t index_batch = 0; index_batch < bn_num_batches; index_batch++) {
            uint16_t index = index_feature * bn_num_batches + index_batch;
            float bn_negation = (bn_input_features[index] - bn_means[index_feature]);
            bn_variances[index_feature] += bn_negation * bn_negation;
        }
    }
    for (uint16_t index_feature = 0; index_feature < bn_num_features; index_feature++) {
        bn_variances[index_feature] /= bn_num_batches;
    }

    // Apply normalization, scale, and shift
    #pragma omp parallel for schedule(dynamic)
    for (uint16_t index_feature = 0; index_feature < bn_num_features; index_feature++) {
        for (uint16_t index_batch = 0; index_batch < bn_num_batches; index_batch++) {
            uint16_t index = index_feature * bn_num_batches + index_batch;
            float bn_normalized_value = (bn_input_features[index] - bn_means[index_feature]) / sqrt(bn_variances[index_feature] + epsilon);
            bn_output_features[index] = bn_gamma[index_feature] * bn_normalized_value + bn_beta[index_feature];
        }
    }
}
