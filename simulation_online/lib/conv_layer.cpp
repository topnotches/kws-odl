#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include "conv_layer.hpp"

#include <stdint.h>
#include <stdio.h>
#include <iostream>
#include "layer_analyzer.hpp"
extern layer_analyzer conv_analyzer;

void conv_layer_float(float *conv_input_features, float *conv_output_features, float *conv_kernel_weights, float *conv_kernel_biases,
                            const uint16_t conv_input_width, const uint16_t conv_input_height, const uint16_t conv_input_depth,
                            const uint8_t conv_stride, const uint8_t conv_kernel_width, const uint8_t conv_kernel_height,
                            const uint8_t output_feats, const uint8_t conv_batch_size,
                            const uint8_t pad_top, const uint8_t pad_bottom, const uint8_t pad_left, const uint8_t pad_right) {

    int32_t conv_batch_input_offset = 0;
    int32_t conv_batch_output_offset = 0;
    const int32_t conv_feature_size = conv_input_width * conv_input_height;
    const int32_t conv_width_limit = (conv_input_width - conv_kernel_width + pad_left + pad_right) / conv_stride + 1;
    const int32_t conv_height_limit = (conv_input_height - conv_kernel_height + pad_top + pad_bottom) / conv_stride + 1;
    const int32_t conv_kernel_depth_offset_multiplier = conv_kernel_width * conv_kernel_height;
    const int32_t conv_total_kernel_select_offset_multiplier = conv_width_limit * conv_height_limit;
    const int32_t conv_batch_input_feature_points_per_batch = conv_input_width * conv_input_height * conv_input_depth;
    const int32_t conv_batch_output_feature_points_per_batch = conv_width_limit * conv_height_limit * output_feats;

    for (int32_t index_batch = 0; index_batch < conv_batch_size; index_batch++) {
        
#if DO_LAYER_ANALYSIS
#else
        //#pragma omp parallel for //collapse(3) schedule(dynamic)
#endif
        for (int32_t index_kernel = 0; index_kernel < output_feats; index_kernel++) {
            for (int32_t index_row = 0; index_row < conv_height_limit; index_row++) {
                for (int32_t index_column = 0; index_column < conv_width_limit; index_column++) {

                    float conv_sum = 0;
                    int32_t conv_output_feature_select_offset = index_kernel * conv_total_kernel_select_offset_multiplier;
                    int32_t conv_kernel_select_offset = index_kernel * conv_kernel_depth_offset_multiplier * conv_input_depth;
                    int32_t conv_input_index_ud = index_row * conv_stride;
                    int32_t conv_input_index_lr = index_column * conv_stride;

                    for (int32_t index_feature = 0; index_feature < conv_input_depth; index_feature++) {
                        int32_t conv_feature_offset = index_feature * conv_feature_size;
                        int32_t conv_kernel_depth_offset = index_feature * conv_kernel_depth_offset_multiplier;

                        for (int32_t conv_index_kernel_height = 0; conv_index_kernel_height < conv_kernel_height; conv_index_kernel_height++) {
                            int32_t conv_kernel_height_offset = conv_index_kernel_height * conv_kernel_width;

                            for (int32_t conv_index_kernel_width = 0; conv_index_kernel_width < conv_kernel_width; conv_index_kernel_width++) {

                                int32_t conv_kernel_full_offset = conv_kernel_depth_offset + conv_kernel_height_offset + conv_index_kernel_width;
 
                                int16_t conv_input_row = conv_input_index_ud + conv_index_kernel_height - pad_top;
                                int16_t conv_input_col = conv_input_index_lr + conv_index_kernel_width - pad_left;
                                int32_t conv_input_full_offset = conv_feature_offset + conv_input_row * conv_input_width + conv_input_col + conv_batch_input_offset;
                                
                                if (conv_input_row >= 0 && conv_input_row < conv_input_height &&
                                    conv_input_col >= 0 && conv_input_col < conv_input_width) {       // padding

                                    conv_sum += conv_kernel_weights[conv_kernel_select_offset + conv_kernel_full_offset]*conv_input_features[conv_input_full_offset];
                                    conv_analyzer.incr_loads();
                                    conv_analyzer.incr_loads();
                                    conv_analyzer.incr_additions();
                                    conv_analyzer.incr_multiplications();
                                }
                            }
                        }
                    }
                    conv_output_features[conv_batch_output_offset + conv_output_feature_select_offset + index_row * conv_width_limit + index_column] = conv_sum + conv_kernel_biases[index_kernel];
                    conv_analyzer.incr_stores();
                    conv_analyzer.incr_additions();
                }
            }
        }
        conv_batch_output_offset += conv_batch_output_feature_points_per_batch;
        conv_batch_input_offset += conv_batch_input_feature_points_per_batch;
    }
    
}

void conv_layer_fixed(int32_t *conv_input_features, int32_t *conv_output_features, int32_t *conv_kernel_weights, int32_t *conv_kernel_biases,
    const uint16_t conv_input_width, const uint16_t conv_input_height, const uint16_t conv_input_depth,
    const uint8_t conv_stride, const uint8_t conv_kernel_width, const uint8_t conv_kernel_height,
    const uint8_t output_feats, const uint8_t conv_batch_size,
    const uint8_t pad_top, const uint8_t pad_bottom, const uint8_t pad_left, const uint8_t pad_right, const float rescale_value) {

    int32_t conv_batch_input_offset = 0;
    int32_t conv_batch_output_offset = 0;
    const int32_t conv_feature_size = conv_input_width * conv_input_height;
    const int32_t conv_width_limit = (conv_input_width - conv_kernel_width + pad_left + pad_right) / conv_stride + 1;
    const int32_t conv_height_limit = (conv_input_height - conv_kernel_height + pad_top + pad_bottom) / conv_stride + 1;
    const int32_t conv_kernel_depth_offset_multiplier = conv_kernel_width * conv_kernel_height;
    const int32_t conv_total_kernel_select_offset_multiplier = conv_width_limit * conv_height_limit;
    const int32_t conv_batch_input_feature_points_per_batch = conv_input_width * conv_input_height * conv_input_depth;
    const int32_t conv_batch_output_feature_points_per_batch = conv_width_limit * conv_height_limit * output_feats;

    for (int32_t index_batch = 0; index_batch < conv_batch_size; index_batch++) {

    #if DO_LAYER_ANALYSIS
    #else
        //#pragma omp parallel for //collapse(3) schedule(dynamic)
    #endif
        for (int32_t index_kernel = 0; index_kernel < output_feats; index_kernel++) {
            for (int32_t index_row = 0; index_row < conv_height_limit; index_row++) {
                for (int32_t index_column = 0; index_column < conv_width_limit; index_column++) {

                    int32_t conv_sum = 0;
                    int32_t conv_output_feature_select_offset = index_kernel * conv_total_kernel_select_offset_multiplier;
                    int32_t conv_kernel_select_offset = index_kernel * conv_kernel_depth_offset_multiplier * conv_input_depth;
                    int32_t conv_input_index_ud = index_row * conv_stride;
                    int32_t conv_input_index_lr = index_column * conv_stride;

                    for (int32_t index_feature = 0; index_feature < conv_input_depth; index_feature++) {
                    int32_t conv_feature_offset = index_feature * conv_feature_size;
                    int32_t conv_kernel_depth_offset = index_feature * conv_kernel_depth_offset_multiplier;

                    for (int32_t conv_index_kernel_height = 0; conv_index_kernel_height < conv_kernel_height; conv_index_kernel_height++) {
                        int32_t conv_kernel_height_offset = conv_index_kernel_height * conv_kernel_width;

                        for (int32_t conv_index_kernel_width = 0; conv_index_kernel_width < conv_kernel_width; conv_index_kernel_width++) {

                            int32_t conv_kernel_full_offset = conv_kernel_depth_offset + conv_kernel_height_offset + conv_index_kernel_width;

                            int16_t conv_input_row = conv_input_index_ud + conv_index_kernel_height - pad_top;
                            int16_t conv_input_col = conv_input_index_lr + conv_index_kernel_width - pad_left;
                            int32_t conv_input_full_offset = conv_feature_offset + conv_input_row * conv_input_width + conv_input_col + conv_batch_input_offset;
                            
                            if (conv_input_row >= 0 && conv_input_row < conv_input_height &&
                                conv_input_col >= 0 && conv_input_col < conv_input_width) {       // padding

                                conv_sum += conv_kernel_weights[conv_kernel_select_offset + conv_kernel_full_offset]*conv_input_features[conv_input_full_offset];
                                conv_analyzer.incr_loads();
                                conv_analyzer.incr_loads();
                                conv_analyzer.incr_additions();
                                conv_analyzer.incr_multiplications();
                            }
                        }
                    }
                    }
                    conv_output_features[conv_batch_output_offset + conv_output_feature_select_offset + index_row * conv_width_limit + index_column] = conv_sum + conv_kernel_biases[index_kernel];
                    conv_analyzer.incr_stores();
                    conv_analyzer.incr_additions();
                }
            }
        }
        conv_batch_output_offset += conv_batch_output_feature_points_per_batch;
        conv_batch_input_offset += conv_batch_input_feature_points_per_batch;
    }

}
