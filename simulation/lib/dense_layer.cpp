#include <stdint.h>
#include <stdio.h>
#include <string.h> // For memset
#include "dense_layer.hpp"


void dense_layer(float *dense_input, float *dense_output, float *dense_weights, float *dense_biases,
              const uint16_t input_size, const uint16_t dense_output_size, const uint16_t dense_batch_size) {
    memset(dense_output, 0, dense_batch_size * dense_output_size * sizeof(float));

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (uint16_t index_batch = 0; index_batch < dense_batch_size; index_batch++) {
        for (uint16_t index_output = 0; index_output < dense_output_size; index_output++) {
            uint16_t batch_offset_input = index_batch * input_size;
            uint16_t batch_offset_output = index_batch * dense_output_size;

            for (uint16_t index_input = 0; index_input < input_size; index_input++) {
                dense_output[batch_offset_output + index_output] += 
                    dense_input[batch_offset_input + index_input] * 
                    dense_weights[index_output * input_size + index_input];
            }
            dense_output[batch_offset_output + index_output] += dense_biases[index_output];
        }
    }
}
