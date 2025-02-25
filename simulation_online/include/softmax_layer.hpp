#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include <stdint.h>

void softmax_layer_float(const float* softmax_inputs, float* softmax_outputs, const uint8_t softmax_batch_size, const uint8_t softmax_num_labels);
void softmax_layer_backward_float(const float* softmax_outputs, 
                            const float* true_labels, 
                            float* softmax_gradients, 
                            const uint8_t softmax_batch_size, 
                            const uint8_t softmax_num_labels);
void softmax_layer_fixed(const int32_t* softmax_inputs, int32_t* softmax_outputs, const uint8_t softmax_batch_size, const uint8_t softmax_num_labels, const float rescale_value, const uint8_t activation_bits);
void softmax_layer_backward_fixed(const int32_t* softmax_outputs, 
                            const int32_t* true_labels, 
                            int32_t* softmax_gradients, 
                            const uint8_t softmax_batch_size, 
                            const uint8_t softmax_num_labels,
                            const float rescale_value);
#endif
