#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include <stdint.h>

void relu_layer_float(const float *relu_input_features, float *relu_output_features, const int relu_width, const int relu_height, const int relu_depth);

void relu_layer_fixed(const int32_t *relu_input_features, int32_t *relu_output_features, const int relu_width, const int relu_height, const int relu_depth, const double rescale_value, const uint8_t activation_bits);

#endif
