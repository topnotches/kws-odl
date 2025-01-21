#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "batch_norm_layer.hpp"

void batch_norm_sequential(float *bn_input_features, float *bn_output_features,
                           float *bn_gamma, float *bn_beta,
                           float *running_mean, float *running_variance,
                           const uint16_t bn_num_features, const uint16_t bn_num_channels, const uint16_t bn_num_batches) {

    const float epsilon = 1e-5; // Small constant for numerical stability
    const float momentum = 0.1f; // Momentum value

    float bn_means[bn_num_channels];
    float bn_variances[bn_num_channels];
    
    // Initialize means and variances to 0
    memset(bn_means, 0, sizeof(float) * bn_num_channels);
    memset(bn_variances, 0, sizeof(float) * bn_num_channels);

    // Calculate feature means
    #pragma omp parallel for reduction(+:bn_means[:bn_num_features]) schedule(dynamic)
    for (uint16_t index_batch = 0; index_batch < bn_num_batches; index_batch++) {
        for (uint16_t index_channel = 0; index_channel < bn_num_channels; index_channel++) {
            for (uint16_t index_feature = 0; index_feature < bn_num_features; index_feature++) {
                uint16_t io_index = index_batch * bn_num_channels * bn_num_features + index_channel * bn_num_features + index_feature;
                bn_means[index_channel] += bn_input_features[io_index];
            }
        }
    }
    for (uint16_t index_channel = 0; index_channel < bn_num_channels; index_channel++) {
        bn_means[index_channel] /= bn_num_batches * bn_num_features;
    }

    // Calculate feature variances
    #pragma omp parallel for reduction(+:bn_variances[:bn_num_features]) schedule(dynamic)
    for (uint16_t index_batch = 0; index_batch < bn_num_batches; index_batch++) {
        for (uint16_t index_channel = 0; index_channel < bn_num_channels; index_channel++) {
            for (uint16_t index_feature = 0; index_feature < bn_num_features; index_feature++) {
                uint16_t io_index = index_batch * bn_num_channels * bn_num_features + index_channel * bn_num_features + index_feature;
                float bn_negation = (bn_input_features[io_index] - bn_means[index_channel]);
                bn_variances[index_channel] += bn_negation * bn_negation;
            }
        }
    }
    for (uint16_t index_channel = 0; index_channel < bn_num_channels; index_channel++) {
        bn_variances[index_channel] /= bn_num_batches * bn_num_features;
    }

    // Apply momentum to update the means and variances
    #pragma omp parallel for schedule(dynamic)
    for (uint16_t index_channel = 0; index_channel < bn_num_channels; index_channel++) {
        bn_means[index_channel] = momentum * bn_means[index_channel] + (1.0f - momentum) * running_mean[index_channel];
        bn_variances[index_channel] = momentum * bn_variances[index_channel] + (1.0f - momentum) * running_variance[index_channel];
    }

    // Apply normalization, affine transformation (scale and shift)
    #pragma omp parallel for schedule(dynamic)
    for (uint16_t index_batch = 0; index_batch < bn_num_batches; index_batch++) {
        for (uint16_t index_channel = 0; index_channel < bn_num_channels; index_channel++) {
            for (uint16_t index_feature = 0; index_feature < bn_num_features; index_feature++) {
                uint16_t io_index = index_batch * bn_num_channels * bn_num_features + index_channel * bn_num_features + index_feature;
                float bn_normalized_value = (bn_input_features[io_index] - running_mean[index_channel]) / sqrt(running_variance[index_channel] + epsilon);
                
                // Apply affine transformation
                bn_output_features[io_index] = bn_gamma[index_channel] * bn_normalized_value + bn_beta[index_channel];
            }
        }
    }
}
