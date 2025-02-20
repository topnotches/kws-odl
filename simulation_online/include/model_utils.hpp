#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H
#include <stdint.h>
#include <vector>
#include <string>
#include <string>
#include "layer.hpp"


std::vector<std::vector<std::string>> load_layers_from_csv_to_vec(const std::string& filePath);

conv_hypr_param_t set_conv_param(uint8_t width, uint8_t height, uint8_t stride, uint8_t count, uint8_t tp, uint8_t bp, uint8_t lp, uint8_t rp);

dense_hypr_param_t set_dense_param(uint32_t output_size, uint32_t input_size = 1);

std::vector<float> str_to_fl32_vec(std::string string_of_floats, std::string delimiter = " ");

std::vector<layer> get_model(std::string model_path, uint8_t batch_size, uint8_t num_classes);

void model_forward(std::vector<layer> &model, std::vector<float> data);

#endif