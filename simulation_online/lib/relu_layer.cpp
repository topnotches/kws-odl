#include <stdint.h>
#include "relu_layer.hpp"

#include "layer_analyzer.hpp"
extern layer_analyzer relu_analyzer;


void relu_layer_float(const float *relu_input_features, float *relu_output_features, const int relu_width, const int relu_height, const int relu_depth) {
    
#if DO_LAYER_ANALYSIS
#else
    //#pragma omp parallel for //collapse(3)
#endif
    for (uint16_t index_depth = 0; index_depth < relu_depth; index_depth++) {
        for (uint16_t index_height = 0; index_height < relu_height; index_height++) {
            for (uint16_t index_width = 0; index_width < relu_width; index_width++) {
                uint32_t relu_depth_offset = index_depth * relu_height * relu_width;
                uint16_t relu_height_offset = index_height * relu_width;
                if (relu_input_features[relu_depth_offset + relu_height_offset + index_width] < 0) {
                    relu_output_features[relu_depth_offset + relu_height_offset + index_width] = 0;
                } else{
                    relu_output_features[relu_depth_offset + relu_height_offset + index_width] = relu_input_features[relu_depth_offset + relu_height_offset + index_width];
                }
                relu_analyzer.incr_loads();
                relu_analyzer.incr_stores();

                // relu_output_features[relu_depth_offset + relu_height_offset + index_width] = (relu_input_features[relu_depth_offset + relu_height_offset + index_width] < 0) ? 0 : relu_input_features[relu_depth_offset + relu_height_offset + index_width];
            }
        }
    }
}

void relu_layer_fixed(const int32_t *relu_input_features, int32_t *relu_output_features, const int relu_width, const int relu_height, const int relu_depth, const float rescale_value) {
    
#if DO_LAYER_ANALYSIS
#else
    //#pragma omp parallel for //collapse(3)
#endif
    for (uint16_t index_depth = 0; index_depth < relu_depth; index_depth++) {
        for (uint16_t index_height = 0; index_height < relu_height; index_height++) {
            for (uint16_t index_width = 0; index_width < relu_width; index_width++) {
                uint32_t relu_depth_offset = index_depth * relu_height * relu_width;
                uint16_t relu_height_offset = index_height * relu_width;
                if (relu_input_features[relu_depth_offset + relu_height_offset + index_width] < 0) {
                    relu_output_features[relu_depth_offset + relu_height_offset + index_width] = 0;
                } else{
                    relu_output_features[relu_depth_offset + relu_height_offset + index_width] = relu_input_features[relu_depth_offset + relu_height_offset + index_width];
                }
                relu_analyzer.incr_loads();
                relu_analyzer.incr_stores();

                // relu_output_features[relu_depth_offset + relu_height_offset + index_width] = (relu_input_features[relu_depth_offset + relu_height_offset + index_width] < 0) ? 0 : relu_input_features[relu_depth_offset + relu_height_offset + index_width];
            }
        }
    }
}