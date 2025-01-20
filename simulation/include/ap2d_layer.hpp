#ifndef AVERAGE_POOL_2D_H
#define AVERAGE_POOL_2D_H

#include <stdint.h>

void ap2d_layer_sequential(float *ap2d_inputs, float *ap2d_outputs,
                                const uint16_t ap2d_input_width, const uint16_t ap2d_input_height, const uint16_t ap2d_input_depth,
                                const uint8_t ap2d_stride, const uint8_t ap2d_kernel_width, const uint8_t ap2d_kernel_height, const uint8_t ap2d_batch_size,
                                const uint8_t ap2d_pad_top, const uint8_t ap2d_pad_bottom, const uint8_t ap2d_pad_left, const uint8_t ap2d_pad_right);


#endif