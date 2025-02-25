#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <stdint.h>


void dense_layer_float(float *dense_input, float *dense_output, float *dense_weights, float *dense_biases,
    const uint16_t input_size, const uint16_t dense_output_size, const uint16_t dense_batch_size);
void dense_layer_backward_float(float* dense_grad_input, 
                      const float* dense_grad_output, 
                      const float* dense_weights,
                      const uint16_t dense_input_size, 
                      const uint16_t dense_output_size, 
                      const uint16_t dense_batch_size);
void dense_layer_fixed(int32_t *dense_input, int32_t *dense_output, int32_t *dense_weights, int32_t *dense_biases,
                const uint16_t input_size, const uint16_t dense_output_size, const uint16_t dense_batch_size, const float rescale_value, const uint8_t activation_bits);
void dense_layer_backward_fixed(int32_t* dense_grad_input, 
                                const int32_t* dense_grad_output, 
                                const int32_t* dense_weights,
                                const uint16_t dense_input_size, 
                                const uint16_t dense_output_size, 
                                const uint16_t dense_batch_size,
                                const float rescale_value);
#endif
