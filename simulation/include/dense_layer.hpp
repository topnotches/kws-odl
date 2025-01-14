#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <stdint.h>


void dense_layer(float *dense_input, float *dense_output, float *dense_weights, float *dense_biases,
              const uint16_t input_size, const uint16_t dense_output_size, const uint16_t dense_batch_size);

#endif
