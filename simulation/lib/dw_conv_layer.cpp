#include <stdint.h>
#include <stdio.h>
#include "dw_conv_layer.hpp"

void dw_conv_layer_sequential(float *dw_input_features, float *dw_output_features, float *dw_kernel_weights, float *dw_kernel_biases,
                            const uint16_t dw_input_width, const uint16_t dw_input_height, const uint16_t dw_input_depth,
                            const uint8_t dw_stride, const uint8_t dw_kernel_width, const uint8_t dw_kernel_height, const uint8_t dw_batch_size,
                            const uint8_t dw_pad_top, const uint8_t dw_pad_bottom, const uint8_t dw_pad_left, const uint8_t dw_pad_right) {

    uint16_t dw_batch_input_offset = 0;
    uint16_t dw_batch_output_offset = 0;
    
    const uint16_t dw_feature_size = dw_input_width * dw_input_height;
    const uint16_t dw_width_limit = (dw_input_width - dw_kernel_width + dw_pad_left + dw_pad_right) / dw_stride + 1;
    const uint16_t dw_height_limit = (dw_input_height - dw_kernel_height + dw_pad_top + dw_pad_bottom) / dw_stride + 1;
    const uint16_t dw_kernel_size = dw_kernel_width * dw_kernel_height;
    const uint16_t dw_total_kernel_select_offset_multiplier = dw_width_limit * dw_height_limit;
    const uint16_t dw_batch_input_feature_points_per_batch = dw_input_width * dw_input_width * dw_input_depth;
    const uint16_t dw_batch_output_feature_points_per_batch = dw_width_limit * dw_height_limit * dw_input_depth;

    for (uint16_t index_batch = 0; index_batch < dw_batch_size; index_batch++) { // batch
                                // //printf("new batch\n");
        #pragma omp parallel for collapse(3) schedule(dynamic) // till i collapse
        for (uint16_t index_layer = 0; index_layer < dw_input_depth; index_layer++) { // layer
            for (uint16_t index_row = 0; index_row < dw_height_limit; index_row++) { // out
                for (uint16_t index_column = 0; index_column < dw_width_limit ; index_column++) {

                    float dw_sum = 0;
                    uint16_t dw_output_feature_select_offset = index_layer * dw_total_kernel_select_offset_multiplier;
                    uint16_t dw_feature_offset = index_layer * dw_feature_size;
                    uint16_t dw_kernel_size_offset = index_layer * dw_kernel_size;
                    uint16_t dw_input_index_ud = index_row * dw_stride;
                    uint16_t dw_input_index_lr = index_column * dw_stride;
                    for (uint16_t index_kernel_height = 0; index_kernel_height < dw_kernel_height; index_kernel_height++) {
                        for (uint16_t index_kernel_width = 0; index_kernel_width < dw_kernel_width; index_kernel_width++) {
                            
                            int16_t dw_input_row = dw_input_index_ud + index_kernel_height - dw_pad_top;
                            int16_t dw_input_col = dw_input_index_lr + index_kernel_width - dw_pad_left;
                            
                            uint16_t dw_kernel_height_offset = index_kernel_height * dw_kernel_width;
                            uint16_t dw_kernel_full_offset = dw_kernel_size_offset + dw_kernel_height_offset + index_kernel_width;
                            uint16_t dw_input_full_offset = dw_feature_offset + dw_input_row * dw_input_width + dw_input_col + dw_batch_input_offset;

                            if (dw_input_row >= 0 && dw_input_row < dw_input_height &&
                                dw_input_col >= 0 && dw_input_col < dw_input_width) {       // padding
                                //printf("row: %d,  ", dw_input_row);
                                //printf("col: %d, ", dw_input_col);

                               // if (dw_batch_output_offset + dw_output_feature_select_offset + index_row * dw_width_limit + index_column==0)
                               //     printf("input index %d, kernel index %d, index layer %d, kernel value %f, input value %f, bias value %f\n",  dw_input_full_offset, dw_kernel_full_offset, index_layer, dw_kernel_weights[dw_kernel_full_offset], dw_input_features[dw_input_full_offset], dw_kernel_biases[index_layer]);

                                dw_sum += dw_input_features[dw_input_full_offset] * dw_kernel_weights[dw_kernel_full_offset];
                            }
                        }
                    }
                    dw_output_features[dw_batch_output_offset + dw_output_feature_select_offset + index_row * dw_width_limit + index_column] = dw_sum + dw_kernel_biases[index_layer];
                }

            }
        }
        dw_batch_output_offset += dw_batch_output_feature_points_per_batch;
        dw_batch_input_offset += dw_batch_input_feature_points_per_batch;
    }
}
