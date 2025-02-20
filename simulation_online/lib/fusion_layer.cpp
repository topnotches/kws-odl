
#include "fusion_layer.hpp"
#include <math.h>
#include <cstring>
#include <iostream>
#include "layer_analyzer.hpp"
extern layer_analyzer fusion_fw_analyzer;
extern layer_analyzer fusion_bw_analyzer;

void fusion_mult_sequential(const float* fusion_input, float* fusion_output, const float* fusion_embeddings, const uint8_t fusion_features, const uint8_t fusion_batch_size) {
    
#if DO_LAYER_ANALYSIS
#else
    //#pragma omp parallel for //collapse(2) schedule(dynamic)
#endif
    for (uint8_t index_batch = 0; index_batch < fusion_batch_size; index_batch++) {
        for (uint8_t index_feature = 0; index_feature < fusion_features; index_feature++) {
            fusion_output[index_batch * fusion_features + index_feature] = fusion_input[index_batch * fusion_features + index_feature] * fusion_embeddings[index_feature];
            fusion_fw_analyzer.incr_loads();
            fusion_fw_analyzer.incr_loads();
            fusion_fw_analyzer.incr_stores();
            fusion_fw_analyzer.incr_multiplications();
        }
    }
}
void fusion_mult_backward_sequential(float* fusion_grad_input, 
                                    const float* fusion_grad_output, 
                                    const float* fusion_embeddings, 
                                    const uint8_t fusion_features, 
                                    const uint8_t fusion_batch_size) {
    
    memset(fusion_grad_input, 0, fusion_batch_size * fusion_features * sizeof(float));

#if DO_LAYER_ANALYSIS
#else
    //#pragma omp parallel for //collapse(2) schedule(dynamic)
#endif
    for (uint8_t index_batch = 0; index_batch < fusion_batch_size; index_batch++) {
        for (uint8_t index_feature = 0; index_feature < fusion_features; index_feature++) {
            fusion_grad_input[index_batch * fusion_features + index_feature] = 
                fusion_grad_output[index_batch * fusion_features + index_feature] * fusion_embeddings[index_feature];
            fusion_bw_analyzer.incr_loads();
            fusion_bw_analyzer.incr_loads();
            fusion_bw_analyzer.incr_stores();
            fusion_bw_analyzer.incr_multiplications();
        }
    }
}
