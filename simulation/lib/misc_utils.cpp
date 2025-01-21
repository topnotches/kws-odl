#include "misc_utils.hpp"
#include <string>
#include <sstream>
#include <vector>
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


std::vector<std::vector<std::string>> load_layers_from_csv_to_vec(const std::string& filePath) {
    std::ifstream file(filePath);
    std::string line;
    std::vector<std::vector<std::string>> csv_data;
    int current_row = 0;
    

    while (getline(file, line)) {
        if(current_row>0) {

            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row_data, row_data_raw;
            std::string start0_ss = "\"[";
            std::string stop0_ss = "]\"";
            std::string start1_ss = "[";
            std::string stop1_ss = "]";
            std::string name;
            std::string size = "";
            std::string params = "";
            uint8_t clean_state = 0;

            while (getline(ss, cell, ',')) {
                row_data_raw.push_back(cell);
            }

            for (auto string_cell : row_data_raw) {
                switch (clean_state)
                {
                case 0:{
                    name = string_cell;

                    clean_state++;
                    break;
                }
                case 1:{

                    
                    std::string::size_type start0_find = string_cell.find(start0_ss);
                    if (start0_find != std::string::npos)
                        string_cell.erase(start0_find, start0_ss.length());
                    std::string::size_type start1_find = string_cell.find(start1_ss);
                    if (start1_find != std::string::npos)
                        string_cell.erase(start1_find, start1_ss.length());

                    std::string::size_type stop0_find = string_cell.find(stop0_ss);
                    if (stop0_find != std::string::npos) {
                        string_cell.erase(stop0_find, stop0_ss.length());
                        clean_state++;
                    }
                    std::string::size_type stop1_find = string_cell.find(stop1_ss);
                    if (stop1_find != std::string::npos) {
                        string_cell.erase(stop1_find, stop1_ss.length());
                        clean_state++;
                    }
                    size = size + string_cell;

                    break;
                }
                case 2:{

                    std::string::size_type start_find = string_cell.find(start0_ss);
                    std::string::size_type stop_find = string_cell.find(stop0_ss);
                    
                    if (start_find != std::string::npos)
                        string_cell.erase(start_find, start0_ss.length());

                    if (stop_find != std::string::npos) {
                        string_cell.erase(stop_find, stop0_ss.length());
                        clean_state++;
                    }
                    params = params + string_cell;

                    break;
                }
                
                default: {

                    break;
                }
                }
            }

            row_data.push_back(name);
            row_data.push_back(size);
            row_data.push_back(params + " ");

            csv_data.push_back(row_data);
        }
        current_row++;
    }

    file.close();
    return csv_data;
}

std::vector<float> str_to_fl32_vec(std::string string_of_floats, std::string delimiter) {
    std::vector<float> float_vector;

    auto position = string_of_floats.find(delimiter);


    while (position != std::string::npos) {

        
        float_vector.push_back(std::stof(string_of_floats.substr(0, position)));
        string_of_floats.erase(0, position + delimiter.length());

        position = string_of_floats.find(delimiter);
    }
    return float_vector;
}
