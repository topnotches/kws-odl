
#include "fusion_layer.hpp"
#include <math.h>
#include <cstring>
#include <iostream>

void fusion_mult_sequential(const float* fusion_input, float* fusion_output, const float* fusion_embeddings, const uint8_t fusion_features, const uint8_t fusion_batch_size) {
    
    for (uint8_t index_batch = 0; index_batch < fusion_batch_size; index_batch++) {
        for (uint8_t index_feature = 0; index_feature < fusion_features; index_feature++) {
            fusion_output[index_batch * fusion_features + index_feature] = fusion_input[index_batch * fusion_features + index_feature] * fusion_embeddings[index_feature];
            
        }
    }
}
void fusion_mult_backward_sequential(float* fusion_grad_input, 
                                    const float* fusion_grad_output, 
                                    const float* fusion_embeddings, 
                                    const uint8_t fusion_features, 
                                    const uint8_t fusion_batch_size) {
    
    memset(fusion_grad_input, 0, fusion_batch_size * fusion_features * sizeof(float));

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (uint8_t index_batch = 0; index_batch < fusion_batch_size; index_batch++) {
        for (uint8_t index_feature = 0; index_feature < fusion_features; index_feature++) {
            fusion_grad_input[index_batch * fusion_features + index_feature] = 
                fusion_grad_output[index_batch * fusion_features + index_feature] * fusion_embeddings[index_feature];
        }
    }
}
