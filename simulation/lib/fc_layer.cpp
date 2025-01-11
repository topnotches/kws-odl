#include <stdint.h>
#include <stdio.h>
#include "fc_layer.hpp"




void fc_layer(float *fc_input, float *fc_output, float *fc_weights, float *fc_biases, const uint16_t input_size, const uint16_t output_size, const uint16_t batch_size) {
    for (uint8_t index_batch = 0; index_batch < batch_size; index_batch++) {
        uint16_t batch_offset_input = index_batch * input_size;
        uint16_t batch_offset_output = index_batch * output_size;
        for (uint8_t index_output = 0; index_output < output_size; index_output++) {
            for (uint8_t index_input = 0; index_input < input_size; index_input++) {
                fc_output[batch_offset_output + index_output] += fc_input[batch_offset_input + index_input] * fc_weights[index_output * input_size + index_input] + fc_biases[index_output * input_size + index_input]; 
                // printf("index in %d, ", batch_offset_input + index_input);
                // printf("index out %d, ", batch_offset_output + index_output);
                // printf("index param %d \n", index_output * input_size + index_input);
            }
        }
    }
}