#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include <stdint.h>

void softmax_layer_sequential(const float* softmax_inputs, float* softmax_outputs, const uint8_t softmax_batch_size, const uint8_t softmax_num_labels);
void softmax_layer_backward_sequential(const float* softmax_outputs, 
                            const float* true_labels, 
                            float* softmax_gradients, 
                            const uint8_t softmax_batch_size, 
                            const uint8_t softmax_num_labels);
#endif
