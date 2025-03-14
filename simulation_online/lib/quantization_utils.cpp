#include "quantization_utils.hpp"
#include <cmath>
#include <cstdint>
#include <stdio.h>

#include <string>
#include <iostream>
#include "quantization_utils.hpp"

#include "defs.hpp"
#include <algorithm>

int32_t requantize_shift(int32_t large_number, const double rescale_value, const uint8_t activation_bits, const bool use_relu) {
    int32_t my_return = 0;
#if USE_SHIFT_REQUANT

    float shift = log2(rescale_value);

    if (shift < 0.0f) {
        // my_return = static_cast<int32_t>(rescale_value*static_cast<double>(large_number));
        uint16_t shift_round = static_cast<uint16_t>(ceil(fabs(shift)+0.1f));
        float fraction = fabs(shift) - (long)fabs(shift);
        float rescale_value_second_order = 1.0f - fraction;
        float shift_second_order = log2(rescale_value_second_order);
        uint16_t shift_round_second_order  = static_cast<uint16_t>(ceil(fabs(shift_second_order)+0.1f));
        float fraction_second_order = fabs(shift_second_order) - (long)fabs(shift_second_order);
        float rescale_value_third_order = 1.0f - fraction_second_order;
        float shift_third_order = log2(rescale_value_third_order);
        uint16_t shift_round_third_order  = static_cast<uint16_t>(ceil(fabs(shift_third_order)+0.1f));
        
        int32_t result_first_order = 0;
        int32_t result_second_order = 0;
        int32_t result_third_order = 0;

        result_first_order = (large_number >> shift_round);

#if DO_SECOND_ORDER_SHIFT_REQUANT
        result_second_order = (large_number >> shift_round + shift_round_second_order);
#endif

#if DO_THIRD_ORDER_SHIFT_REQUANT
        result_third_order = (large_number >> shift_round + shift_round_second_order + shift_round_third_order);
#endif

        int32_t shift_result = result_first_order + result_second_order + result_third_order;
        /*
        std::cout << "float value" << std::endl;
        std::cout << "my_return............................." << my_return << std::endl;
        std::cout << "my_return............................." << shift_result << std::endl;
        std::cout << "shift................................." << shift << std::endl;
        std::cout << "shift_round..........................." << shift_round << std::endl;
        std::cout << "fraction.............................." << fraction << std::endl;
        std::cout << "rescale_value_second_order............" << rescale_value_second_order << std::endl;
        std::cout << "shift_second_order...................." << shift_second_order << std::endl;
        std::cout << "shift_round_second_order.............." << shift_round_second_order << std::endl;
        std::cout << "fraction_second_order................." << fraction_second_order << std::endl;
        std::cout << "rescale_value_third_order............." << rescale_value_third_order << std::endl;
        std::cout << "shift_third_order....................." << shift_third_order << std::endl;
        std::cout << "shift_round_third_order..............." << shift_round_third_order << std::endl;
        std::cout << "result_first_order...................." << result_first_order << std::endl;
        std::cout << "result_second_order..................." << result_second_order << std::endl;
        std::cout << "result_third_order.........................." << result_third_order << std::endl;
        */
        my_return = shift_result;
    } else {
        //my_return = static_cast<int32_t>(rescale_value*static_cast<double>(large_number));
        uint16_t shift_round = static_cast<uint16_t>(floor(fabs(shift)+0.1f));
        float fraction = fabs(shift) - (long)fabs(shift);
        float rescale_value_second_order =  fraction;
        float shift_second_order = log2(rescale_value_second_order);
        uint16_t shift_round_second_order  = static_cast<uint16_t>(floor(fabs(shift_second_order)+0.1f));
        float fraction_second_order = fabs(shift_second_order) - (long)fabs(shift_second_order);
        float rescale_value_third_order =  fraction_second_order;
        float shift_third_order = log2(rescale_value_third_order);
        uint16_t shift_round_third_order  = static_cast<uint16_t>(floor(fabs(shift_third_order)+0.1f));
        int32_t result_first_order = 0;
        int32_t result_second_order = 0;
        int32_t result_third_order = 0;
        
        result_first_order  = (large_number << shift_round);

#if DO_THIRD_ORDER_SHIFT_REQUANT
        if (shift_round != 0 && shift_round_second_order != 0)
            result_second_order = (large_number << shift_round - shift_round_second_order);
#endif
    
#if DO_THIRD_ORDER_SHIFT_REQUANT
        if (shift_round != 0 && shift_round_second_order != 0 && shift_round_second_order != 0)
            result_third_order        = (large_number << shift_round - shift_round_second_order - shift_round_third_order);
#endif
        int32_t shift_result = result_first_order + result_second_order + result_third_order;
        
        my_return = shift_result;
    }
#else
    my_return = static_cast<int32_t>(rescale_value*static_cast<double>(large_number));

    if (use_relu && my_return < 0) {
        my_return = 0;
    }

#endif

    int32_t min;
    int32_t max;

    if (use_relu ) {
        min = 0;
        max = static_cast<int32_t>(pow(2, activation_bits)+0.5);
    } else {
        min = -static_cast<int32_t>(pow(2, activation_bits-1)+0.5);
        max = static_cast<int32_t>(pow(2, activation_bits-1)+0.5)-1;
    }
    my_return = std::clamp(my_return, min, max);
    
    return my_return;
}
int32_t requantize_shift_old(int32_t large_number, const double rescale_value, const uint8_t activation_bits, const bool use_relu) {
    double distance = 99999999999;
    double temp_distance = 0;
    int8_t shift = 0;
    int8_t one = 1;
    double interp_rescale_value = 1;
    int32_t shifted_large_number;
    
    double lsb_size =interp_rescale_value / (int)(pow(2, activation_bits+1) + 0.5);

    // find shiftness
    while (1) {
        if (rescale_value > 1) {
            temp_distance = (rescale_value - interp_rescale_value * (int)(pow(2, shift) + 0.5));
        } else {
            temp_distance = (rescale_value - interp_rescale_value / (int)(pow(2, shift) + 0.5));
        }
        //std::cout << std::to_string(shift) << std::endl;
        //std::cout << std::to_string(abs(temp_distance)) << std::endl;
        
        if (abs(temp_distance) < abs(distance)) {
            distance = temp_distance;
        } else {
            shift--;
        //std::cout << std::to_string(shift) << std::endl;
        //std::cout << std::to_string(abs(rescale_value - interp_rescale_value / (2.0 * shift))) << std::endl;
        //std::cout << std::to_string(abs(interp_rescale_value / (2.0 * shift))) << std::endl;

            break;
        }
        shift++;
    }

    // Round to Odd because I'm too lazy for RTNE...
    
    if (large_number < 0 && use_relu) { // relu :)
        shifted_large_number = 0;
    } else {
        if (rescale_value > 1) {
            shifted_large_number = large_number << shift;
        } else {
            shifted_large_number = large_number >> shift;
            int remainder = large_number & ((1 << shift) - 1);
            int round_bit = (large_number >> (shift - 1)) & 1;
            int sticky_bit = remainder != 0;
            if (round_bit && (shifted_large_number % 2 != 0 || sticky_bit)) {
                shifted_large_number += 1;
            }

        }
    
        if ((shifted_large_number >> activation_bits) > 0) {
            shifted_large_number = (int)(pow(2, activation_bits - (1-use_relu)) +0.5)-1;
        } else if (shifted_large_number < -(int)(pow(2, activation_bits- (1-use_relu)) +0.5)-1) {
/*
std::cout << -(int)(pow(2, activation_bits- (1-use_relu)) +0.5)-1 << std::endl;
std::cout << -(int)(pow(2, activation_bits- (1-use_relu)) +0.5)-1 << std::endl;
std::cout << -(int)(pow(2, activation_bits- (1-use_relu)) +0.5)-1 << std::endl;
std::cout << -(int)(pow(2, activation_bits- (1-use_relu)) +0.5)-1 << std::endl;
std::cout << -(int)(pow(2, activation_bits- (1-use_relu)) +0.5)-1 << std::endl;
std::cout << shifted_large_number << std::endl;
std::cout << shifted_large_number << std::endl;
std::cout << shifted_large_number << std::endl;
std::cout << shifted_large_number << std::endl;
std::cout << shifted_large_number << std::endl;
std::cout << shifted_large_number << std::endl;
*/
            shifted_large_number = -(int)(pow(2, activation_bits- (1-use_relu)) +0.5)-1;
        }
    }
    
    return shifted_large_number;
}

int32_t requantize_shift_sm(int32_t large_number, const double rescale_value, const uint8_t activation_bits, const bool use_relu) {
    double distance = 99999999999;
    double temp_distance = 0;
    int8_t shift = 0;
    int8_t one = 1;
    double interp_rescale_value = 1;
    int32_t shifted_large_number;
    
    double lsb_size =interp_rescale_value / (int)(pow(2, activation_bits+1) + 0.5);

    // find shiftness
    while (1) {
        if (rescale_value > 1) {
            temp_distance = (rescale_value - interp_rescale_value * (int)(pow(2, shift) + 0.5));
        } else {
            temp_distance = (rescale_value - interp_rescale_value / (int)(pow(2, shift) + 0.5));
        }
        //std::cout << std::to_string(shift) << std::endl;
        //std::cout << std::to_string(abs(temp_distance)) << std::endl;
        
        if (abs(temp_distance) < abs(distance)) {
            distance = temp_distance;
        } else {
            shift--;
        //std::cout << std::to_string(shift) << std::endl;
        //std::cout << std::to_string(abs(rescale_value - interp_rescale_value / (2.0 * shift))) << std::endl;
        //std::cout << std::to_string(abs(interp_rescale_value / (2.0 * shift))) << std::endl;

            break;
        }
        shift++;
    }
    //std::cout << std::to_string(rescale_value) << std::endl;

    // Round to Odd because I'm too lazy for RTNE...
    
    if (large_number < 0 && use_relu) { // relu :)
        shifted_large_number = 0;
    } else {
        if (rescale_value > 1) {
            shifted_large_number = large_number << shift;
        } else {
            shifted_large_number = large_number >> shift;
            int remainder = large_number & ((1 << shift) - 1);
            int round_bit = (large_number >> (shift - 1)) & 1;
            int sticky_bit = remainder != 0;
            if (round_bit && (shifted_large_number % 2 != 0 || sticky_bit)) {
                shifted_large_number += 1;
            }

        }
    
        if ((shifted_large_number >> activation_bits) > 0) {
            shifted_large_number = (int)(pow(2, activation_bits - (1-use_relu)) +0.5)-1;
        } else if ((shifted_large_number >> activation_bits) > (int)(pow(2, activation_bits) +0.5)-1) {
            
            shifted_large_number = -(int)(pow(2, activation_bits- (1-use_relu)) +0.5)-1;
        }
    }
    return shifted_large_number;
}

/*
rescale = 126
approx = 125

dist = 126 - 125 = 1 


*/