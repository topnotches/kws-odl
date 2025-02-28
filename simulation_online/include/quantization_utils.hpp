#ifndef QUANTIZATION_UTILS_H
#define QUANTIZATION_UTILS_H

#include <stdint.h>

int32_t requantize_shift(int32_t large_number, const double rescale_value, const uint8_t activation_bits, const bool use_relu = true);
int32_t requantize_shift_sm(int32_t large_number, const double rescale_value, const uint8_t activation_bits, const bool use_relu);

#endif