#ifndef FUSION_LAYER_H
#define FUSION_LAYER_H

#include <stdint.h>

void fusion_mult_sequential(const float* fusion_input, float* fusion_output, const float* fusion_embeddings, const uint8_t fusion_features, const uint8_t fusion_batch_size);

void fusion_mult_backward_sequential(float* fusion_grad_input, 
                                    const float* fusion_grad_output, 
                                    const float* fusion_embeddings, 
                                    const uint8_t fusion_features, 
                                    const uint8_t fusion_batch_size);
#endif
