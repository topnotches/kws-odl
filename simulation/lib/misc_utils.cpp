#include "misc_utils.hpp"
#include <string>
#include <assert.h>



void init_flarr_to_num(float *my_array, const uint16_t my_array_size, const float my_float) {
    for (auto i = 0; i < my_array_size; i++)
        my_array[i] = my_float;
}


void KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN(const char *message) {
    printf("\n");
    printf(message);
    printf("\n");
    assert(1 == 0);
}