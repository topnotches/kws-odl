#include <stdint.h>
#include <stdio.h>
#include "dw_conv_layer.hpp"

void dw_conv_layer_sequential(float *dw_conv_input_features, float *dw_conv_output_features, float *dw_conv_kernel_weights, float *dw_conv_kernel_biases,
                            const uint16_t dw_conv_input_width, const uint16_t dw_conv_input_height, const uint16_t dw_conv_input_depth,
                            const uint8_t dw_conv_stride, const uint8_t dw_conv_kernel_width, const uint8_t dw_conv_kernel_height, const uint8_t dw_conv_batch_size) {

    uint16_t dw_conv_batch_input_offset = 0;
    uint16_t dw_conv_batch_output_offset = 0;
    
    const uint16_t dw_conv_feature_size = dw_conv_input_width * dw_conv_input_height;
    
    const uint16_t dw_conv_width_limit = (dw_conv_input_width - dw_conv_kernel_width) / dw_conv_stride + 1;
    const uint16_t dw_conv_height_limit = (dw_conv_input_height - dw_conv_kernel_height) / dw_conv_stride + 1;

    const uint16_t dw_conv_row_offset_multiplier = dw_conv_input_width * dw_conv_stride;
    const uint16_t dw_conv_kernel_size = dw_conv_kernel_width * dw_conv_kernel_height;

    const uint16_t dw_conv_total_kernel_select_offset_multiplier = dw_conv_width_limit * dw_conv_height_limit;
    
    const uint16_t dw_conv_batch_input_feature_points_per_batch = dw_conv_input_width * dw_conv_input_width * dw_conv_input_depth;
    
    const uint16_t dw_conv_batch_output_feature_points_per_batch = dw_conv_width_limit * dw_conv_height_limit * dw_conv_input_depth;



    for (uint16_t index_batch = 0; index_batch < dw_conv_batch_size; index_batch++) { // batch

        #pragma omp parallel for collapse(3) schedule(dynamic) // till i collapse
        for (uint16_t index_layer = 0; index_layer < dw_conv_input_depth; index_layer++) { // layer
            for (uint16_t index_row = 0; index_row < dw_conv_width_limit; index_row++) { // out
                for (uint16_t index_column = 0; index_column < dw_conv_height_limit; index_column++) {

                    float dw_conv_sum = 0;
                    uint16_t dw_conv_row_offset = index_row * dw_conv_row_offset_multiplier;
                    uint16_t dw_conv_column_offset = index_column * dw_conv_stride;
                    uint16_t dw_conv_output_feature_select_offset = index_layer * dw_conv_total_kernel_select_offset_multiplier;
                    uint16_t dw_conv_feature_offset = index_layer * dw_conv_feature_size;
                    uint16_t dw_conv_kernel_size_offset = index_layer * dw_conv_kernel_size;

                    for (uint16_t index_kernel_height = 0; index_kernel_height < dw_conv_kernel_height; index_kernel_height++) {
                        for (uint16_t index_kernel_width = 0; index_kernel_width < dw_conv_kernel_width; index_kernel_width++) {
                            uint16_t dw_conv_kernel_height_offset = index_kernel_height * dw_conv_kernel_width;

                            uint16_t dw_conv_input_full_offset = dw_conv_feature_offset + dw_conv_row_offset + index_kernel_height * dw_conv_input_width + dw_conv_column_offset + index_kernel_width + dw_conv_batch_input_offset;
                            uint16_t dw_conv_kernel_full_offset = dw_conv_kernel_size_offset + dw_conv_kernel_height_offset + index_kernel_width;

                            dw_conv_sum += dw_conv_input_features[dw_conv_input_full_offset] * dw_conv_kernel_weights[dw_conv_kernel_full_offset] + dw_conv_kernel_biases[dw_conv_kernel_full_offset];

                        }
                    }
                    dw_conv_output_features[dw_conv_batch_output_offset + dw_conv_output_feature_select_offset + index_row * dw_conv_width_limit + index_column] = dw_conv_sum;
                }

            }
        }
        dw_conv_batch_output_offset += dw_conv_batch_output_feature_points_per_batch;
        dw_conv_batch_input_offset += dw_conv_batch_input_feature_points_per_batch;
    }
}
