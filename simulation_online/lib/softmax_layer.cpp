#include "softmax_layer.hpp"
#include <math.h>
#include <iostream>
#include <string>
#include "layer_analyzer.hpp"
#include "quantization_utils.hpp"
extern layer_analyzer softmax_fw_analyzer;
extern layer_analyzer softmax_bw_analyzer;

void softmax_layer_float(const float* softmax_inputs, float* softmax_outputs, const uint8_t softmax_batch_size, const uint8_t softmax_num_labels) {
    for (uint8_t index_batch = 0; index_batch < softmax_batch_size; index_batch++) {
        float denominator = 0.0f;
        
        for (uint8_t index_input = 0; index_input < softmax_num_labels; index_input++) {
            denominator += exp(softmax_inputs[index_input + index_batch * softmax_num_labels]);
            softmax_fw_analyzer.incr_loads();
            softmax_fw_analyzer.incr_loads();
            softmax_fw_analyzer.incr_additions();
        }
        //std::cout << "oasiej" << denominator << std::endl;
        for (uint8_t index_output = 0; index_output < softmax_num_labels; index_output++) {
            softmax_outputs[index_output + index_batch * softmax_num_labels] =
                exp(softmax_inputs[index_output + index_batch * softmax_num_labels]) / denominator;
                softmax_fw_analyzer.incr_loads();
                softmax_fw_analyzer.incr_loads();
                softmax_fw_analyzer.incr_stores();
                softmax_fw_analyzer.incr_additions();
        }
    }
}
void softmax_layer_backward_float(const float* softmax_outputs, 
                            const float* true_labels, 
                            float* softmax_gradients, 
                            const uint8_t softmax_batch_size, 
                            const uint8_t softmax_num_labels) {
    for (uint8_t index_batch = 0; index_batch < softmax_batch_size; index_batch++) {
        for (uint8_t index_label = 0; index_label < softmax_num_labels; index_label++) {
            softmax_gradients[index_label + index_batch * softmax_num_labels] = 
                softmax_outputs[index_label + index_batch * softmax_num_labels] - 
                true_labels[index_label + index_batch * softmax_num_labels];
            softmax_bw_analyzer.incr_loads();
            softmax_bw_analyzer.incr_loads();
            softmax_bw_analyzer.incr_stores();
            softmax_bw_analyzer.incr_additions();
        }
    }
}
void softmax_layer_fixed(const int32_t* softmax_inputs, int32_t* softmax_outputs, const uint8_t softmax_batch_size, const uint8_t softmax_num_labels, const double rescale_value, const uint8_t activation_bits) {

    for (uint8_t index_batch = 0; index_batch < softmax_batch_size; index_batch++) {
        int32_t denominator = 0;

        for (uint8_t index_input = 0; index_input < softmax_num_labels; index_input++) {
            auto e_out = (int32_t)(exp((double)requantize_shift(softmax_inputs[index_input + index_batch * softmax_num_labels], rescale_value, activation_bits, false)));
            if (e_out > pow(2,activation_bits)-1) {
                e_out = pow(2,activation_bits)-1;
            }
            denominator += e_out;

        }
        for (uint8_t index_output = 0; index_output < softmax_num_labels; index_output++) {


            auto e_out = (exp((double)requantize_shift(softmax_inputs[index_output + index_batch * softmax_num_labels], rescale_value, activation_bits, false)));
            if (e_out > pow(2,activation_bits)-1) {
                e_out = pow(2,activation_bits)-1;
            }
            softmax_outputs[index_output + index_batch * softmax_num_labels] = (int32_t)(255 * e_out / denominator) ;
           // std::cout << "softmax_outputs[index_output + index_batch * softmax_num_labels]"  << std::endl;
           // std::cout << softmax_outputs[index_output + index_batch * softmax_num_labels]  << std::endl;
        }
    }
}
void softmax_layer_backward_fixed(const int32_t* softmax_outputs, 
                            const int32_t* true_labels, 
                            int32_t* softmax_gradients, 
                            const uint8_t softmax_batch_size, 
                            const uint8_t softmax_num_labels,
                            const double rescale_value,
                            const uint8_t input_activation_bits) {
    for (uint8_t index_batch = 0; index_batch < softmax_batch_size; index_batch++) {
        for (uint8_t index_label = 0; index_label < softmax_num_labels; index_label++) {
            softmax_gradients[index_label + index_batch * softmax_num_labels] =
                requantize_shift(
                softmax_outputs[index_label + index_batch * softmax_num_labels] - (1<<8)*true_labels[index_label + index_batch * softmax_num_labels]-true_labels[index_label + index_batch * softmax_num_labels],
                1.0f,
                input_activation_bits,
                false);
            softmax_bw_analyzer.incr_loads();
            softmax_bw_analyzer.incr_loads();
            softmax_bw_analyzer.incr_stores();
            softmax_bw_analyzer.incr_additions();
        }
    }
}