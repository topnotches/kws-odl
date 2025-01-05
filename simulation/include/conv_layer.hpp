
#include <stdint.h>


void conv_layer_sequential(float *conv_input_features, float *conv_output_features, float *conv_kernels,
                            const uint16_t conv_input_width, const uint16_t conv_input_height, const uint16_t conv_input_depth,
                            const uint8_t conv_stride, const uint8_t conv_kernel_width, const uint8_t conv_kernel_height, const uint8_t output_feats);