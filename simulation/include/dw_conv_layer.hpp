#ifndef DW_CONV_H
#define DW_CONV_H

#include <stdint.h>

void dw_conv_layer_sequential(float *dw_input_features, float *dw_output_features, float *dw_kernel_weights, float *dw_kernel_biases,
                            const uint16_t dw_input_width, const uint16_t dw_input_height, const uint16_t dw_input_depth,
                            const uint8_t dw_stride, const uint8_t dw_kernel_width, const uint8_t dw_kernel_height, const uint8_t dw_batch_size,
                            const uint8_t dw_pad_top, const uint8_t dw_pad_bottom, const uint8_t dw_pad_left, const uint8_t dw_pad_right);


#endif