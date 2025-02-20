#include <stdint.h>
#include <stdio.h>
#include "avgpool2d_layer.hpp"

void avgpool2d_layer_sequential(float *avgpool2d_inputs, float *avgpool2d_outputs,
                                const uint16_t avgpool2d_input_width, const uint16_t avgpool2d_input_height, const uint16_t avgpool2d_input_depth,
                                const uint8_t avgpool2d_stride, const uint8_t avgpool2d_kernel_width, const uint8_t avgpool2d_kernel_height, const uint8_t avgpool2d_batch_size,
                                const uint8_t avgpool2d_pad_top, const uint8_t avgpool2d_pad_bottom, const uint8_t avgpool2d_pad_left, const uint8_t avgpool2d_pad_right) {

    uint32_t avgpool2d_batch_input_offset = 0;
    uint32_t avgpool2d_batch_output_offset = 0;
    
    const uint32_t avgpool2d_feature_size = avgpool2d_input_width * avgpool2d_input_height;
    const uint32_t avgpool2d_width_limit = (avgpool2d_input_width - avgpool2d_kernel_width + avgpool2d_pad_left + avgpool2d_pad_right) / avgpool2d_stride + 1;
    const uint32_t avgpool2d_height_limit = (avgpool2d_input_height - avgpool2d_kernel_height + avgpool2d_pad_top + avgpool2d_pad_bottom) / avgpool2d_stride + 1;
    const uint32_t avgpool2d_batch_input_feature_points_per_batch = avgpool2d_input_width * avgpool2d_input_height * avgpool2d_input_depth;
    const uint32_t avgpool2d_batch_output_feature_points_per_batch = avgpool2d_width_limit * avgpool2d_height_limit * avgpool2d_input_depth;

    for (uint32_t index_batch = 0; index_batch < avgpool2d_batch_size; index_batch++) { // batch

#if DO_LAYER_ANALYSIS
#else
        //#pragma omp parallel for //collapse(3)
#endif
        for (uint32_t index_layer = 0; index_layer < avgpool2d_input_depth; index_layer++) { // layer
            for (uint32_t index_row = 0; index_row < avgpool2d_width_limit; index_row++) { // out
                for (uint32_t index_column = 0; index_column < avgpool2d_height_limit; index_column++) {

                    float avgpool2d_sum = 0;
                    uint32_t avgpool2d_count = 0;
                    uint32_t avgpool2d_feature_offset = index_layer * avgpool2d_feature_size;
                    uint32_t avgpool2d_input_index_ud = index_row * avgpool2d_stride;
                    uint32_t avgpool2d_input_index_lr = index_column * avgpool2d_stride;

                    for (uint32_t index_kernel_height = 0; index_kernel_height < avgpool2d_kernel_height; index_kernel_height++) {
                        for (uint32_t index_kernel_width = 0; index_kernel_width < avgpool2d_kernel_width; index_kernel_width++) {
                            
                            int16_t avgpool2d_input_row = avgpool2d_input_index_ud + index_kernel_height - avgpool2d_pad_top;
                            int16_t avgpool2d_input_col = avgpool2d_input_index_lr + index_kernel_width - avgpool2d_pad_left;
                            
                            if (avgpool2d_input_row >= 0 && avgpool2d_input_row < avgpool2d_input_height &&
                                avgpool2d_input_col >= 0 && avgpool2d_input_col < avgpool2d_input_width) {  // padding
                                uint32_t avgpool2d_input_full_offset = avgpool2d_feature_offset + avgpool2d_input_row * avgpool2d_input_width + avgpool2d_input_col + avgpool2d_batch_input_offset;
                                avgpool2d_sum += avgpool2d_inputs[avgpool2d_input_full_offset];
                                avgpool2d_count++;
                            }
                        }
                    }
                    avgpool2d_outputs[avgpool2d_batch_output_offset + index_layer * avgpool2d_width_limit * avgpool2d_height_limit + index_row * avgpool2d_width_limit + index_column] = avgpool2d_sum / avgpool2d_count;
                }
            }
        }
        avgpool2d_batch_output_offset += avgpool2d_batch_output_feature_points_per_batch;
        avgpool2d_batch_input_offset += avgpool2d_batch_input_feature_points_per_batch;
    }
}
