#ifndef AVERAGE_POOL_2D_H
#define AVERAGE_POOL_2D_H

#include <stdint.h>

void avgpool2d_layer_float(float *avgpool2d_inputs, float *avgpool2d_outputs,
                                const uint16_t avgpool2d_input_width, const uint16_t avgpool2d_input_height, const uint16_t avgpool2d_input_depth,
                                const uint8_t avgpool2d_stride, const uint8_t avgpool2d_kernel_width, const uint8_t avgpool2d_kernel_height, const uint8_t avgpool2d_batch_size,
                                const uint8_t avgpool2d_pad_top, const uint8_t avgpool2d_pad_bottom, const uint8_t avgpool2d_pad_left, const uint8_t avgpool2d_pad_right);

void avgpool2d_layer_fixed(int32_t *avgpool2d_inputs, int32_t *avgpool2d_outputs,
                                const uint16_t avgpool2d_input_width, const uint16_t avgpool2d_input_height, const uint16_t avgpool2d_input_depth,
                                const uint8_t avgpool2d_stride, const uint8_t avgpool2d_kernel_width, const uint8_t avgpool2d_kernel_height, const uint8_t avgpool2d_batch_size,
                                const uint8_t avgpool2d_pad_top, const uint8_t avgpool2d_pad_bottom, const uint8_t avgpool2d_pad_left, const uint8_t avgpool2d_pad_right, const double rescale_value, const uint8_t activation_bits);


#endif