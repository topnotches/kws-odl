#include "misc_utils.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <stdio.h>
#include "defs.hpp"
#include <fstream>
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





std::vector<float> load_mffcs_bin(const std::string& filename) {
    std::vector<float> data;
    
    // Open the binary file for reading
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    // Get the file size
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Ensure the file size is a multiple of float size
    if (size % sizeof(float) != 0) {
        std::cerr << "File size is not a multiple of float size!" << std::endl;
        return data;
    }

    // Resize the vector to hold the float data
    size_t numFloats = size / sizeof(float);
    data.resize(numFloats);

    // Read the binary file into the float vector
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        std::cerr << "Error reading file: " << filename << std::endl;
        data.clear();
    }

    return data;
}