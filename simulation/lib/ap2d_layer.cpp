#include <stdint.h>
#include <stdio.h>
#include "ap2d_layer.hpp"

void ap2d_layer_sequential(float *ap2d_inputs, float *ap2d_outputs,
                                const uint16_t ap2d_input_width, const uint16_t ap2d_input_height, const uint16_t ap2d_input_depth,
                                const uint8_t ap2d_stride, const uint8_t ap2d_kernel_width, const uint8_t ap2d_kernel_height, const uint8_t ap2d_batch_size,
                                const uint8_t ap2d_pad_top, const uint8_t ap2d_pad_bottom, const uint8_t ap2d_pad_left, const uint8_t ap2d_pad_right) {

    uint16_t ap2d_batch_input_offset = 0;
    uint16_t ap2d_batch_output_offset = 0;
    
    const uint16_t ap2d_feature_size = ap2d_input_width * ap2d_input_height;
    const uint16_t ap2d_width_limit = (ap2d_input_width - ap2d_kernel_width + ap2d_pad_left + ap2d_pad_right) / ap2d_stride + 1;
    const uint16_t ap2d_height_limit = (ap2d_input_height - ap2d_kernel_height + ap2d_pad_top + ap2d_pad_bottom) / ap2d_stride + 1;
    const uint16_t ap2d_batch_input_feature_points_per_batch = ap2d_input_width * ap2d_input_height * ap2d_input_depth;
    const uint16_t ap2d_batch_output_feature_points_per_batch = ap2d_width_limit * ap2d_height_limit * ap2d_input_depth;

    for (uint16_t index_batch = 0; index_batch < ap2d_batch_size; index_batch++) { // batch
        #pragma omp parallel for collapse(3) schedule(dynamic)
        for (uint16_t index_layer = 0; index_layer < ap2d_input_depth; index_layer++) { // layer
            for (uint16_t index_row = 0; index_row < ap2d_width_limit; index_row++) { // out
                for (uint16_t index_column = 0; index_column < ap2d_height_limit; index_column++) {

                    float ap2d_sum = 0;
                    uint16_t ap2d_count = 0;
                    uint16_t ap2d_feature_offset = index_layer * ap2d_feature_size;
                    uint16_t ap2d_input_index_ud = index_row * ap2d_stride;
                    uint16_t ap2d_input_index_lr = index_column * ap2d_stride;

                    for (uint16_t index_kernel_height = 0; index_kernel_height < ap2d_kernel_height; index_kernel_height++) {
                        for (uint16_t index_kernel_width = 0; index_kernel_width < ap2d_kernel_width; index_kernel_width++) {
                            
                            int16_t ap2d_input_row = ap2d_input_index_ud + index_kernel_height - ap2d_pad_top;
                            int16_t ap2d_input_col = ap2d_input_index_lr + index_kernel_width - ap2d_pad_left;
                            
                            if (ap2d_input_row >= 0 && ap2d_input_row < ap2d_input_height &&
                                ap2d_input_col >= 0 && ap2d_input_col < ap2d_input_width) {  // padding
                                uint16_t ap2d_input_full_offset = ap2d_feature_offset + ap2d_input_row * ap2d_input_width + ap2d_input_col + ap2d_batch_input_offset;
                                ap2d_sum += ap2d_inputs[ap2d_input_full_offset];
                                ap2d_count++;
                            }
                        }
                    }
                    ap2d_outputs[ap2d_batch_output_offset + index_layer * ap2d_width_limit * ap2d_height_limit + index_row * ap2d_width_limit + index_column] = ap2d_sum / ap2d_count;
                }
            }
        }
        ap2d_batch_output_offset += ap2d_batch_output_feature_points_per_batch;
        ap2d_batch_input_offset += ap2d_batch_input_feature_points_per_batch;
    }
}
