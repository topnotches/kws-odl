#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "batch_norm_layer.hpp"

void batch_norm_sequential(float *bn_input_features, float *bn_output_features,
                           float *running_mean, float *running_variance,
                           float *bn_gamma, float *bn_beta,
                           
                           const uint16_t bn_num_features, const uint16_t bn_num_channels, const uint16_t bn_num_batches) {

    const float epsilon = 1e-5; // Small constant for numerical stability
    const float momentum = 0.1f; // Momentum value
    /*
    float running_mean[64]={95.730064,-88.965675,-13.862235,110.13989,116.39144,
                            -95.57904,-244.60301,25.611305,351.99844,-182.29115,
                            -46.31411,-117.96838,25.619434,116.014946,269.98825,
                            -114.918396,-17.930262,-138.54442,-306.2582,-228.34201,
                            57.63616,29.995892,20.992685,3.7501512,49.773712,
                            245.63559,85.47211,87.91832,-15.907516,77.2689,
                            -148.26947,-41.95629,-122.0923,-71.934845,54.547745,
                            53.310734,164.33199,103.17618,84.011734,-70.19945,
                            109.84035,60.467285,68.87156,-7.603095,-257.14478,
                            -38.686428,-245.98773,38.95627,164.25287,42.70832,
                            70.71899,154.85965,195.25336,97.5785,-199.06502,
                            -5.0245423,281.71185,-162.04172,85.17744,23.656775,
                            28.451738,105.06433,-63.867752,6.1714616};
    float running_variance[64]={1009.3773,3223.6665,1200.3853,2725.181,875.8155,1460.6572,
                                2763.2874,1019.9004,6183.5303,801.8437,345.42575,640.5284,
                                1500.0997,4553.4805,6427.971,816.908,3905.2805,1436.0745,
                                2979.3457,3032.2747,988.6049,2275.079,1941.5804,3030.2405,
                                477.09308,3397.785,3483.9434,473.47043,1356.3973,2713.9583,
                                2123.4963,4090.2004,648.7113,918.6136,1359.7157,1200.8126,
                                2481.819,4606.497,1132.7461,242.47707,3169.2,826.52673,
                                3891.553,281.6628,1501.816,665.25,1355.9979,1429.1229,
                                4500.571,329.4808,1456.3563,799.7965,1742.5712,1761.6469,
                                1402.7675,473.95032,5195.6055,1140.547,2349.7307,639.79517,
                                645.08,1528.618,1299.8351,1077.0165};
    */
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
