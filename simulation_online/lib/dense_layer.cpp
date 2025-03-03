#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "quantization_utils.hpp"
#include "dense_layer.hpp"
#include "layer_analyzer.hpp"
extern layer_analyzer dense_fw_analyzer;
extern layer_analyzer dense_bw_analyzer;


void dense_layer_float(float *dense_input, float *dense_output, float *dense_weights, float *dense_biases,
              const uint16_t input_size, const uint16_t dense_output_size, const uint16_t dense_batch_size) {
    memset(dense_output, 0, dense_batch_size * dense_output_size * sizeof(float));

    for (uint16_t index_batch = 0; index_batch < dense_batch_size; index_batch++) {
        for (uint16_t index_output = 0; index_output < dense_output_size; index_output++) {
            uint16_t batch_offset_input = index_batch * input_size;
            uint16_t batch_offset_output = index_batch * dense_output_size;
            for (uint16_t index_input = 0; index_input < input_size; index_input++) {
                dense_output[batch_offset_output + index_output] += 
                    dense_input[batch_offset_input + index_input] * 
                    dense_weights[index_output * input_size + index_input];
                    dense_fw_analyzer.incr_loads();
                    dense_fw_analyzer.incr_loads();
                    dense_fw_analyzer.incr_additions();
                    dense_fw_analyzer.incr_multiplications();
            }
            dense_output[batch_offset_output + index_output] += dense_biases[index_output];
        }
    }
}
void dense_layer_backward_float(float* dense_grad_input, 
                                const float* dense_grad_output, 
                                const float* dense_weights,
                                const uint16_t dense_input_size, 
                                const uint16_t dense_output_size, 
                                const uint16_t dense_batch_size) {

    // Initialize gradient input
    memset(dense_grad_input, 0, dense_batch_size * dense_input_size * sizeof(float));

    for (uint16_t index_batch = 0; index_batch < dense_batch_size; index_batch++) {

        for (uint16_t index_output = 0; index_output < dense_output_size; index_output++) {
            // Compute gradient w.r.t. inputs
            for (uint16_t index_input = 0; index_input < dense_input_size; index_input++) {
                dense_grad_input[index_batch * dense_input_size + index_input] += 
                    dense_grad_output[index_batch * dense_output_size + index_output] * 
                    dense_weights[index_output * dense_input_size + index_input];
                    dense_bw_analyzer.incr_loads();
                    dense_bw_analyzer.incr_loads();
                    dense_bw_analyzer.incr_additions();
                    dense_bw_analyzer.incr_multiplications();

            }
        }
    }
}



void dense_layer_fixed(int32_t *dense_input, int32_t *dense_output, int32_t *dense_weights, int32_t *dense_biases,
    const uint16_t input_size, const uint16_t dense_output_size, const uint16_t dense_batch_size, const double rescale_value, const uint8_t activation_bits) {
memset(dense_output, 0, dense_batch_size * dense_output_size * sizeof(int32_t));

    for (uint16_t index_batch = 0; index_batch < dense_batch_size; index_batch++) {
        for (uint16_t index_output = 0; index_output < dense_output_size; index_output++) {
            uint16_t batch_offset_input = index_batch * input_size;
            uint16_t batch_offset_output = index_batch * dense_output_size;
            for (uint16_t index_input = 0; index_input < input_size; index_input++) {
                dense_output[batch_offset_output + index_output] += 
                    dense_input[batch_offset_input + index_input] * 
                    dense_weights[index_output * input_size + index_input];
                    dense_fw_analyzer.incr_loads();
                    dense_fw_analyzer.incr_loads();
                    dense_fw_analyzer.incr_additions();
                    dense_fw_analyzer.incr_multiplications();
            }
            dense_output[batch_offset_output + index_output] = requantize_shift(dense_output[batch_offset_output + index_output] + dense_biases[index_output], rescale_value, activation_bits, false);
        }
    }
}
void dense_layer_backward_fixed(int32_t* dense_grad_input, 
                      const int32_t* dense_grad_output, 
                      const int32_t* dense_weights,
                      const uint16_t dense_input_size, 
                      const uint16_t dense_output_size, 
                      const uint16_t dense_batch_size,
                      const double rescale_value,
                      const uint8_t input_activation_bits) {

    // Initialize gradient input
    memset(dense_grad_input, 0, dense_batch_size * dense_input_size * sizeof(int32_t));

    for (uint16_t index_batch = 0; index_batch < dense_batch_size; index_batch++) {
        for (uint16_t index_output = 0; index_output < dense_output_size; index_output++) {
            // Compute gradient w.r.t. inputs
            for (uint16_t index_input = 0; index_input < dense_input_size; index_input++) {
           /*
           std::cout << std::to_string(                    dense_grad_output[index_batch * dense_output_size + index_output] * 
           dense_weights[index_output * dense_input_size + index_input]) << std::endl;
           std::cout << rescale_value << std::endl;
           */

                dense_grad_input[index_batch * dense_input_size + index_input] += 
                    dense_grad_output[index_batch * dense_output_size + index_output] * 
                    dense_weights[index_output * dense_input_size + index_input];
                    dense_bw_analyzer.incr_loads();
                    dense_bw_analyzer.incr_loads();
                    dense_bw_analyzer.incr_additions();
                    dense_bw_analyzer.incr_multiplications();

            }
            //std::cout << "oajfoiajf" << std::endl;
        }
    }

    // Requantize
    for (uint16_t index_batch = 0; index_batch < dense_batch_size; index_batch++) {
        for (uint16_t index_input = 0; index_input < dense_input_size; index_input++) {
            dense_grad_input[index_batch * dense_input_size + index_input] = requantize_shift(dense_grad_input[index_batch * dense_input_size + index_input], rescale_value, input_activation_bits, false); 
            //std::cout << std::to_string(dense_grad_input[index_batch * dense_input_size + index_input]) << std::endl;
        }
    }
}
