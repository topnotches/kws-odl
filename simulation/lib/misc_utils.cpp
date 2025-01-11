#include "misc_utils.hpp"




void init_flarr_to_num(float *my_array, const uint16_t my_array_size, const float my_float) {
    for (auto i = 0; i < my_array_size; i++)
        my_array[i] = my_float;
}
