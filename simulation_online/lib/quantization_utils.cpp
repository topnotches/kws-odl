#include "quantization_utils.hpp"
#include <cmath>
#include <cstdint>
#include <stdio.h>

#include <string>
#include <iostream>
#include "quantization_utils.hpp"
#include <math.h>

int32_t requantize_shift(int32_t large_number, const float rescale_value, const uint8_t activation_bits) {
    float distance = 99999999999;
    float temp_distance = 0;
    int8_t shift = 0;
    int8_t one = 1;
    float interp_rescale_value = 1;
    int32_t shifted_large_number;
    
    // find shiftness
    while (1) {
        if (rescale_value > 1) {
            temp_distance = abs(rescale_value - interp_rescale_value * (int)(pow(2, shift) +0.5));
        } else {
            temp_distance = abs(rescale_value - interp_rescale_value / (int)(pow(2, shift) +0.5));
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
    if (rescale_value > 1) {
        shifted_large_number = large_number << shift;
    } else {
        // L = (large_number >> shift) % 2;
        // G = (large_number >> shift + 1) % 2;
        // R = (large_number >> shift + 2) % 2;
        // T = 
        shifted_large_number = large_number >> shift;
        if (shifted_large_number % 2 == 0) {
            shifted_large_number += 1;
        }
    }
    if ((shifted_large_number >> activation_bits) > 0) {
        shifted_large_number = (int)(pow(2, activation_bits) +0.5)-1;
    }
    if (shifted_large_number < 0) { // relu :)
        shifted_large_number = 0;
    }
    return shifted_large_number;
}