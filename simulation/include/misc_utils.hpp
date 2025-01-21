#ifndef MISC_UTILS_H
#define MISC_UTILS_H

#include <stdint.h>
#include <vector>
#include <string>
#include "defs.hpp"


void init_flarr_to_num(float *my_array, const uint16_t my_array_size, const float my_float);
void KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN(const char *message);


std::vector<std::vector<std::string>> load_layers_from_csv_to_vec(const std::string& filePath);

std::vector<float> str_to_fl32_vec(std::string string_of_floats, std::string delimiter = " ");

std::vector<float> load_mffcs_bin(const std::string& filename);

conv_hypr_param_t set_conv_param(uint8_t width, uint8_t height, uint8_t stride, uint8_t count, uint8_t tp, uint8_t bp, uint8_t lp, uint8_t rp);

dense_hypr_param_t set_dense_param(uint32_t output_size, uint32_t input_size = 1);

#endif