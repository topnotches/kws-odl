#include "softmax_layer.hpp"
#include <math.h>
#include <iostream>


void softmax_layer_sequential(const float* softmax_inputs, float* softmax_outputs, const uint8_t softmax_batch_size, const uint8_t softmax_num_labels) {
    for (uint8_t index_batch = 0; index_batch < softmax_batch_size; index_batch++) {
        float denominator = 0.0f;

        for (uint8_t index_input = 0; index_input < softmax_num_labels; index_input++) {
            denominator += exp(softmax_inputs[index_input + index_batch * softmax_num_labels]);
        }
        //std::cout << "oasiej" << denominator << std::endl;
        for (uint8_t index_output = 0; index_output < softmax_num_labels; index_output++) {
            softmax_outputs[index_output + index_batch * softmax_num_labels] =
                exp(softmax_inputs[index_output + index_batch * softmax_num_labels]) / denominator;
        }
    }
}
void softmax_layer_backward_sequential(const float* softmax_outputs, 
                            const float* true_labels, 
                            float* softmax_gradients, 
                            const uint8_t softmax_batch_size, 
                            const uint8_t softmax_num_labels) {
    for (uint8_t index_batch = 0; index_batch < softmax_batch_size; index_batch++) {
        for (uint8_t index_label = 0; index_label < softmax_num_labels; index_label++) {
            softmax_gradients[index_label + index_batch * softmax_num_labels] = 
                softmax_outputs[index_label + index_batch * softmax_num_labels] - 
                true_labels[index_label + index_batch * softmax_num_labels];

        }
    }
}