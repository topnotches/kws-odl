#ifndef BATCH_NORM_LAYER_H
#define BATCH_NORM_LAYER_H

#include <stdint.h>

void batch_norm_sequential(float *bn_input_features, float *bn_output_features,
                           float *bn_gamma, float *bn_beta,
                           float *running_mean, float *running_variance,
                           const uint16_t bn_num_features, const uint16_t bn_num_channels, const uint16_t bn_num_batches);

#endif