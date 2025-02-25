#ifndef BATCH_NORM_LAYER_H
#define BATCH_NORM_LAYER_H

#include <stdint.h>

void batch_norm_float(float *bn_input_features, float *bn_output_features,
    float *bn_gamma, float *bn_beta,
    float *running_mean, float *running_variance,
    const uint16_t bn_num_features, const uint16_t bn_num_channels, const uint16_t bn_num_batches);

void batch_norm_fixed(int32_t *bn_input_features, int32_t *bn_output_features,
    int32_t *bn_gamma, int32_t *bn_beta,
    int32_t *running_mean, int32_t *running_variance,
    const uint16_t bn_num_features, const uint16_t bn_num_channels, const uint16_t bn_num_batches, const float rescale_value, const uint8_t activation_bits);

#endif