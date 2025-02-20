#ifndef MISC_UTILS_H
#define MISC_UTILS_H

#include <stdint.h>
#include <vector>
#include <string>
#include "defs.hpp"


void init_flarr_to_num(float *my_array, const uint16_t my_array_size, const float my_float);
void KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN(const char *message);
std::vector<float> load_mffcs_bin(const std::string& filename);
std::vector<float> int_to_float_onehot(const uint8_t integer, const uint8_t max);

#endif