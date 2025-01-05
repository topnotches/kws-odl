#include <stdint.h>


void batch_norm_sequential(float *bn_input_features, float *bn_output_features, float *bn_gamma,
                        float *bn_beta, const uint16_t bn_num_features, const uint16_t bn_num_batches);