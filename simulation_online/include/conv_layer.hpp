#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <stdint.h>


void conv_layer_float(float *conv_input_features, float *conv_output_features, float *conv_kernel_weights, float *conv_kernel_biases,
                            const uint16_t conv_input_width, const uint16_t conv_input_height, const uint16_t conv_input_depth,
                            const uint8_t conv_stride, const uint8_t conv_kernel_width, const uint8_t conv_kernel_height,
                            const uint8_t output_feats, const uint8_t conv_batch_size,
                            const uint8_t pad_top, const uint8_t pad_bottom, const uint8_t pad_left, const uint8_t pad_right);



void conv_layer_fixed(int32_t *conv_input_features, int32_t *conv_output_features, int32_t *conv_kernel_weights, int32_t *conv_kernel_biases,
                            const uint16_t conv_input_width, const uint16_t conv_input_height, const uint16_t conv_input_depth,
                            const uint8_t conv_stride, const uint8_t conv_kernel_width, const uint8_t conv_kernel_height,
                            const uint8_t output_feats, const uint8_t conv_batch_size,
                            const uint8_t pad_top, const uint8_t pad_bottom, const uint8_t pad_left, const uint8_t pad_right, const float rescale_value);
    
    
#endif