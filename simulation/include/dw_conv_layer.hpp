#ifndef DW_CONV_H
#define DW_CONV_H

#include <stdint.h>


void dw_conv_layer_sequential(float *dw_conv_input_features, float *dw_conv_output_features, float *dw_conv_kernel_weights, float *dw_conv_kernel_biases,
                            const uint16_t dw_conv_input_width, const uint16_t dw_conv_input_height, const uint16_t dw_conv_input_depth,
                            const uint8_t dw_conv_stride, const uint8_t dw_conv_kernel_width, const uint8_t dw_conv_kernel_height, const uint8_t dw_conv_batch_size);


#endif