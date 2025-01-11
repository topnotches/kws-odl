#include <stdint.h>
#include "relu_layer.hpp"



void relu_layer(float *relu_input_features, float *relu_output_features, const int relu_width, const int relu_height, const int relu_depth) {

    for (uint16_t index_depth = 0; index_depth < relu_depth; index_depth++) {
        uint16_t relu_depth_offset = index_depth * relu_height * relu_width;
        for (uint16_t index_height = 0; index_height < relu_height; index_height++) {
            uint16_t relu_height_offset = index_height * relu_width;
            for (uint16_t index_width = 0; index_width < relu_width; index_width++) {
                if (relu_input_features[relu_depth_offset + relu_height_offset + index_width] < 0) {
                    relu_output_features[relu_depth_offset + relu_height_offset + index_width] = 0;
                } else{
                    relu_output_features[relu_depth_offset + relu_height_offset + index_width] = relu_input_features[relu_depth_offset + relu_height_offset + index_width];
                }
                // relu_output_features[relu_depth_offset + relu_height_offset + index_width] = (relu_input_features[relu_depth_offset + relu_height_offset + index_width] < 0) ? 0 : relu_input_features[relu_depth_offset + relu_height_offset + index_width];
            }
        }
    }
}