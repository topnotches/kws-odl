#include <stdint.h>
#include <stdio.h>
#include "conv_layer.hpp"

void conv_layer_sequential(float *conv_input_features, float *conv_output_features, float *conv_kernels,
                            const uint16_t conv_input_width, const uint16_t conv_input_height, const uint16_t conv_input_depth,
                            const uint8_t conv_stride, const uint8_t conv_kernel_width, const uint8_t conv_kernel_height, const uint8_t output_feats) {

    const uint16_t conv_feature_size = conv_input_width * conv_input_height;
    const uint16_t conv_width_limit = (conv_input_width - conv_kernel_width) / conv_stride + 1;
    const uint16_t conv_height_limit = (conv_input_height - conv_kernel_height) / conv_stride + 1;
    const uint16_t conv_row_offset_multiplier = conv_input_width * conv_stride;
    const uint16_t conv_kernel_depth_offset_multiplier = conv_kernel_width * conv_kernel_height;

    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (uint16_t index_kernel = 0; index_kernel < output_feats; index_kernel++) {
        for (uint16_t index_row = 0; index_row < conv_width_limit; index_row++) {
            for (uint16_t index_column = 0; index_column < conv_height_limit; index_column++) {

                float conv_sum = 0;
                uint16_t conv_row_offset = index_row * conv_row_offset_multiplier;
                uint16_t conv_column_offset = index_column * conv_stride;
                uint16_t conv_output_feature_select_offset = index_kernel * conv_width_limit * conv_height_limit;
                uint16_t conv_kernel_select_offset = index_kernel * conv_kernel_depth_offset_multiplier * conv_input_depth;

                for (uint16_t index_feature = 0; index_feature < conv_input_depth; index_feature++) {
                    uint16_t conv_feature_offset = index_feature * conv_feature_size;
                    uint16_t conv_kernel_depth_offset = index_feature * conv_kernel_depth_offset_multiplier;

                    for (uint16_t index_kernel_height = 0; index_kernel_height < conv_kernel_height; index_kernel_height++) {
                        uint16_t conv_kernel_height_offset = index_kernel_height * conv_kernel_width;

                        for (uint16_t index_kernel_width = 0; index_kernel_width < conv_kernel_width; index_kernel_width++) {
                            uint16_t conv_input_full_offset = conv_feature_offset + conv_row_offset + index_kernel_height * conv_input_width + conv_column_offset + index_kernel_width;
                            uint16_t conv_kernel_full_offset = conv_kernel_depth_offset + conv_kernel_height_offset + index_kernel_width;

                            conv_sum += conv_input_features[conv_input_full_offset] * conv_kernels[conv_kernel_select_offset + conv_kernel_full_offset];
                        }
                    }
                }
                conv_output_features[conv_output_feature_select_offset + index_row * conv_width_limit + index_column] = conv_sum;
            }
        }
    }
}
