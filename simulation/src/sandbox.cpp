#include <iostream>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <vector>
#include "misc_utils.hpp"
#include "defs.hpp"




int  main() {
    std::string exported_csv_relative_path_name_for_testing_csv_what_a_ridiculously_long_name_this_is = "./exported_models/export_params_nclass_12.csv";
    
    std::vector<std::vector<std::string>> bobby = load_layers_from_csv_to_vec(exported_csv_relative_path_name_for_testing_csv_what_a_ridiculously_long_name_this_is);
    std::vector<float> my_params = string_to_float_vector(bobby[0][2]);
    //for (float param : my_params) {
    //    printf("%f ", param);
    //}
    /*
    std::cout << bobby[0][0] << "  "<< bobby[0][2] << std::endl;
    std::cout << bobby[1][0] << "  "<< bobby[1][2] << std::endl;
    std::cout << bobby[2][0] << "  "<< bobby[2][2] << std::endl;
    std::cout << bobby[3][0] << "  "<< bobby[3][2] << std::endl;
    std::cout << bobby[4][0] << "  "<< bobby[4][2] << std::endl;
    std::cout << bobby[5][0] << "  "<< bobby[5][2] << std::endl;
    std::cout << bobby[6][0] << "  "<< bobby[6][2] << std::endl;
    std::cout << bobby[7][0] << "  "<< bobby[7][2] << std::endl;
    std::cout << bobby[8][0] << "  "<< bobby[8][2] << std::endl;
    std::cout << bobby[9][0] << "  "<< bobby[9][2] << std::endl;
    std::cout << bobby[10][0] << "  "<< bobby[10][2] << std::endl;
    std::cout << bobby[11][0] << "  "<< bobby[11][2] << std::endl;
    std::cout << bobby[12][0] << "  "<< bobby[12][2] << std::endl;
    std::cout << bobby[13][0] << "  "<< bobby[13][2] << std::endl;
    std::cout << bobby[14][0] << "  "<< bobby[14][2] << std::endl;
    std::cout << bobby[15][0] << "  "<< bobby[15][2] << std::endl;
    std::cout << bobby[16][0] << "  "<< bobby[16][2] << std::endl;
    std::cout << bobby[17][0] << "  "<< bobby[17][2] << std::endl;
    std::cout << bobby[18][0] << "  "<< bobby[18][2] << std::endl;
    std::cout << bobby[19][0] << "  "<< bobby[19][2] << std::endl;
    std::cout << bobby[20][0] << "  "<< bobby[20][2] << std::endl;
    std::cout << bobby[21][0] << "  "<< bobby[21][2] << std::endl;
    std::cout << bobby[22][0] << "  "<< bobby[22][2] << std::endl;
    std::cout << bobby[23][0] << "  "<< bobby[23][2] << std::endl;
    std::cout << bobby[24][0] << "  "<< bobby[24][2] << std::endl;
    std::cout << bobby[25][0] << "  "<< bobby[25][2] << std::endl;
    std::cout << bobby[26][0] << "  "<< bobby[26][2] << std::endl;
    std::cout << bobby[27][0] << "  "<< bobby[27][2] << std::endl;
    std::cout << bobby[28][0] << "  "<< bobby[28][2] << std::endl;
    std::cout << bobby[29][0] << "  "<< bobby[29][2] << std::endl;
    std::cout << bobby[30][0] << "  "<< bobby[30][2] << std::endl;
    std::cout << bobby[31][0] << "  "<< bobby[31][2] << std::endl;
    std::cout << bobby[32][0] << "  "<< bobby[32][2] << std::endl;
    std::cout << bobby[33][0] << "  "<< bobby[33][2] << std::endl;
    std::cout << bobby[34][0] << "  "<< bobby[34][2] << std::endl;
    std::cout << bobby[35][0] << "  "<< bobby[35][2] << std::endl;
    std::cout << bobby[36][0] << "  "<< bobby[36][2] << std::endl;
    std::cout << bobby[37][0] << "  "<< bobby[37][2] << std::endl;
    */
    return 0;
}