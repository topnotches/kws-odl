#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "batch_norm_layer.hpp"
#include "defs.hpp"
#include "layer_analyzer.hpp"

extern layer_analyzer batchnorm_analyzer;

void batch_norm_float(float *bn_input_features, float *bn_output_features,
                           float *bn_gamma, float *bn_beta,
                           float *running_mean, float *running_variance,
                           const uint16_t bn_num_features, const uint16_t bn_num_channels, const uint16_t bn_num_batches) {

    const float epsilon = 1e-5; // Small constant for numerical stability
    const float momentum = 0.0f; // Momentum value

    float bn_means[bn_num_channels];
    float bn_variances[bn_num_channels];
    
    // Initialize means and variances to 0
    memset(bn_means, 0, sizeof(float) * bn_num_channels);
    memset(bn_variances, 0, sizeof(float) * bn_num_channels);

    // Calculate feature means
    
#if DO_LAYER_ANALYSIS
#else
    //#pragma omp parallel for
#endif
    for (uint16_t index_batch = 0; index_batch < bn_num_batches; index_batch++) {
        for (uint16_t index_channel = 0; index_channel < bn_num_channels; index_channel++) {
            float batch_sum = 0.0f;
            for (uint16_t index_feature = 0; index_feature < bn_num_features; index_feature++) {
                uint32_t io_index = index_batch * bn_num_channels * bn_num_features + index_channel * bn_num_features + index_feature;
                batch_sum += bn_input_features[io_index];
                
#if DO_FULL_BATCHNORM_LAYER_ANALYSIS
                batchnorm_analyzer.incr_loads();
                batchnorm_analyzer.incr_additions();
#endif

            }
            
#if DO_LAYER_ANALYSIS
#else
            //#pragma omp atomic
#endif
            bn_means[index_channel] += batch_sum;
#if DO_FULL_BATCHNORM_LAYER_ANALYSIS
            batchnorm_analyzer.incr_loads();
            batchnorm_analyzer.incr_stores();
            batchnorm_analyzer.incr_additions();
#endif

        }
    }

    for (uint16_t index_channel = 0; index_channel < bn_num_channels; index_channel++) {
        bn_means[index_channel] /= (bn_num_batches * bn_num_features);
#if DO_FULL_BATCHNORM_LAYER_ANALYSIS
            batchnorm_analyzer.incr_loads();
            batchnorm_analyzer.incr_stores();
            batchnorm_analyzer.incr_divisions();

#endif


    }

    // Calculate feature variances
    
#if DO_LAYER_ANALYSIS
#else
    //#pragma omp parallel for
#endif
    for (uint16_t index_batch = 0; index_batch < bn_num_batches; index_batch++) {
        for (uint16_t index_channel = 0; index_channel < bn_num_channels; index_channel++) {
            float batch_variance = 0.0f;
            for (uint16_t index_feature = 0; index_feature < bn_num_features; index_feature++) {
                uint32_t io_index = index_batch * bn_num_channels * bn_num_features + index_channel * bn_num_features + index_feature;
                float diff = bn_input_features[io_index] - bn_means[index_channel];
                batch_variance += diff * diff;
#if DO_FULL_BATCHNORM_LAYER_ANALYSIS
                batchnorm_analyzer.incr_loads();
                batchnorm_analyzer.incr_additions();
                batchnorm_analyzer.incr_multiplications();
#endif
            }
            
#if DO_LAYER_ANALYSIS
#else
            //#pragma omp atomic
#endif
            bn_variances[index_channel] += batch_variance;
#if DO_FULL_BATCHNORM_LAYER_ANALYSIS
            batchnorm_analyzer.incr_loads();
            batchnorm_analyzer.incr_stores();
            batchnorm_analyzer.incr_additions();
#endif

        }
    }

    for (uint16_t index_channel = 0; index_channel < bn_num_channels; index_channel++) {
        bn_variances[index_channel] /= (bn_num_batches * bn_num_features);
#if DO_FULL_BATCHNORM_LAYER_ANALYSIS
        batchnorm_analyzer.incr_loads();
        batchnorm_analyzer.incr_stores();
        batchnorm_analyzer.incr_divisions();
#endif

    }

    // Apply momentum to update the means and variances
    
#if DO_LAYER_ANALYSIS
#else
    //#pragma omp parallel for
#endif
    for (uint16_t index_channel = 0; index_channel < bn_num_channels; index_channel++) {
        running_mean[index_channel] = momentum * bn_means[index_channel] + (1.0f - momentum) * running_mean[index_channel];
#if DO_FULL_BATCHNORM_LAYER_ANALYSIS
        batchnorm_analyzer.incr_loads();
        batchnorm_analyzer.incr_loads();
        batchnorm_analyzer.incr_stores();
        batchnorm_analyzer.incr_additions();
        batchnorm_analyzer.incr_additions();
        batchnorm_analyzer.incr_multiplications();
#endif

        running_variance[index_channel] = momentum * bn_variances[index_channel] + (1.0f - momentum) * running_variance[index_channel];
#if DO_FULL_BATCHNORM_LAYER_ANALYSIS
        batchnorm_analyzer.incr_loads();
        batchnorm_analyzer.incr_loads();
        batchnorm_analyzer.incr_stores();
        batchnorm_analyzer.incr_additions();
        batchnorm_analyzer.incr_additions();
        batchnorm_analyzer.incr_multiplications();
#endif
    }

    // Apply normalization and affine transformation (scale and shift)
    
#if DO_LAYER_ANALYSIS
#else
    //#pragma omp parallel for
#endif
    for (uint16_t index_batch = 0; index_batch < bn_num_batches; index_batch++) {
        for (uint16_t index_channel = 0; index_channel < bn_num_channels; index_channel++) {
            for (uint16_t index_feature = 0; index_feature < bn_num_features; index_feature++) {
                uint32_t io_index = index_batch * bn_num_channels * bn_num_features + index_channel * bn_num_features + index_feature;
                float normalized_value = (bn_input_features[io_index] - running_mean[index_channel]) /
                                         sqrt(running_variance[index_channel] + epsilon);
                                         
                batchnorm_analyzer.incr_loads();
                batchnorm_analyzer.incr_loads();
                batchnorm_analyzer.incr_loads();
                batchnorm_analyzer.incr_additions();
                batchnorm_analyzer.incr_additions();
                batchnorm_analyzer.incr_divisions();
                
                // Apply affine transformation
                bn_output_features[io_index] = bn_gamma[index_channel] * normalized_value + bn_beta[index_channel];

                
                batchnorm_analyzer.incr_loads();
                batchnorm_analyzer.incr_loads();
                batchnorm_analyzer.incr_stores();
                batchnorm_analyzer.incr_additions();
                batchnorm_analyzer.incr_multiplications();

            }
        }
    }
}



void batch_norm_fixed(int32_t *bn_input_features, int32_t *bn_output_features,
    int32_t *bn_gamma, int32_t *bn_beta,
    int32_t *running_mean, int32_t *running_variance,
    const uint16_t bn_num_features, const uint16_t bn_num_channels, const uint16_t bn_num_batches, const float rescale_value, const uint8_t activation_bits) {

    for (uint16_t index_batch = 0; index_batch < bn_num_batches; index_batch++) {
        for (uint16_t index_channel = 0; index_channel < bn_num_channels; index_channel++) {
            for (uint16_t index_feature = 0; index_feature < bn_num_features; index_feature++) {
            int32_t io_index = index_batch * bn_num_channels * bn_num_features + index_channel * bn_num_features + index_feature;
            int32_t normalized_value = (bn_input_features[io_index] - running_mean[index_channel]) /
                            sqrt(running_variance[index_channel] + 1); // TODO and FIXME: "... + 1" is a dirty little epsilon replacement trick >;)
                            
            batchnorm_analyzer.incr_loads();
            batchnorm_analyzer.incr_loads();
            batchnorm_analyzer.incr_loads();
            batchnorm_analyzer.incr_additions();
            batchnorm_analyzer.incr_additions();
            batchnorm_analyzer.incr_divisions();

            // Apply affine transformation
            bn_output_features[io_index] = bn_gamma[index_channel] * normalized_value + bn_beta[index_channel];


            batchnorm_analyzer.incr_loads();
            batchnorm_analyzer.incr_loads();
            batchnorm_analyzer.incr_stores();
            batchnorm_analyzer.incr_additions();
            batchnorm_analyzer.incr_multiplications();

            }
        }
    }
}
