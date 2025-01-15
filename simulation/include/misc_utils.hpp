#ifndef MISC_UTILS_H
#define MISC_UTILS_H

#include <stdint.h>
#include <vector>
#include <string>


void init_flarr_to_num(float *my_array, const uint16_t my_array_size, const float my_float);
void KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN(const char *message);


std::vector<std::vector<std::string>> load_layers_from_csv_to_vec(const std::string& filePath);

std::vector<float> string_to_float_vector(std::string string_of_floats, std::string delimiter = " ");

#endif