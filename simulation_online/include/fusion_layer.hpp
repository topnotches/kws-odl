#ifndef FUSION_LAYER_H
#define FUSION_LAYER_H

#include <stdint.h>

void fusion_mult_float(const float* fusion_input, float* fusion_output, const float* fusion_embeddings, const uint8_t fusion_features, const uint8_t fusion_batch_size);

void fusion_mult_backward_float(float* fusion_grad_input, 
                                    const float* fusion_grad_output, 
                                    const float* fusion_embeddings, 
                                    const uint8_t fusion_features, 
                                    const uint8_t fusion_batch_size);

void fusion_mult_fixed(const int32_t* fusion_input, int32_t* fusion_output, const int32_t* fusion_embeddings, const uint8_t fusion_features, const uint8_t fusion_batch_size, const double rescale_value, const uint8_t activation_bits);

void fusion_mult_backward_fixed(int32_t* fusion_grad_input, 
                                    const int32_t* fusion_grad_output, 
                                    const int32_t* fusion_embeddings, 
                                    const uint8_t fusion_features, 
                                    const uint8_t fusion_batch_size,
                                    const double rescale_value,
                                    const uint8_t gradient_bits);
                                        
void fusion_add_fixed(const int32_t* fusion_input, int32_t* fusion_output, const int32_t* fusion_embeddings, const uint8_t fusion_features, const uint8_t fusion_batch_size, const double rescale_value, const uint8_t activation_bits);

void fusion_add_backward_fixed(int32_t* fusion_grad_input, 
                                    const int32_t* fusion_grad_output, 
                                    const int32_t* fusion_embeddings, 
                                    const uint8_t fusion_features, 
                                    const uint8_t fusion_batch_size,
                                    const double rescale_value,
                                    const uint8_t gradient_bits);
                                                                            
#endif