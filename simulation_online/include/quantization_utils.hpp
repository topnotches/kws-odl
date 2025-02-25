#ifndef QUANTIZATION_UTILS_H
#define QUANTIZATION_UTILS_H

#include <stdint.h>

int32_t requantize_shift(int32_t large_number, const float rescale_value, const uint8_t activation_bits);

#endif