#include "layer_analyzer.hpp"
#include "defs.hpp"
#include <iomanip> // Include this header for std::setw

layer_analyzer batchnorm_analyzer("batchnorm");
layer_analyzer conv_analyzer("conv");
layer_analyzer crossloss_analyzer("crossloss");
layer_analyzer dense_fw_analyzer("dense_fw");
layer_analyzer dense_bw_analyzer("dense_bw");
layer_analyzer dw_analyzer("dw");
layer_analyzer fusion_fw_analyzer("fusion_fw");
layer_analyzer fusion_bw_analyzer("fusion_bw");
layer_analyzer relu_analyzer("relu");
layer_analyzer softmax_fw_analyzer("softmax_fw");
layer_analyzer softmax_bw_analyzer("softmax_bw");

layer_analyzer::layer_analyzer(std::string layer_name) {
    this->anal_layer_name = layer_name;
    this->anal_loads = 0;
    this->anal_stores = 0;
    this->anal_additions = 0;
    this->anal_multiplications = 0;
    this->anal_divisions = 0;
    this->anal_intensity = 0;
}
layer_analyzer::~layer_analyzer() {
}

void layer_analyzer::print_stats() {
    calc_intensity();
    std::cout << "Layer: " << std::setw(10)<< this->anal_layer_name
              << "      Loads/Stores: " << std::setw(10) << this->anal_loads + this->anal_stores
              << "      Adds: " << std::setw(10) << this->anal_additions
              << "      Mults: " << std::setw(10) << this->anal_multiplications
              << "      Divs: " << std::setw(10) << this->anal_divisions
              << "      Arith sum: " << std::setw(10) << this->anal_additions + this->anal_multiplications + this->anal_divisions
              << "      Intensity: " << std::setw(10) << this->anal_intensity << std::endl;
}


void layer_analyzer::print_stats_colnames() {
    std::cout << std::setw(15) << "Layer" << std::setw(15)
                << "Loads/Stores" << std::setw(15) 
                << "Adds" << std::setw(15) 
                << "Mults" << std::setw(15) 
                << "Divs" << std::setw(15) 
                << "Arith sum" << std::setw(15) 
                << "Intensity" << std::endl;
}


void layer_analyzer::print_stats_raw() {
    calc_intensity();

    std::cout << std::setw(15) << this->anal_layer_name << std::setw(15)
             << this->anal_loads + this->anal_stores << std::setw(15)
             << this->anal_additions << std::setw(15)
             << this->anal_multiplications << std::setw(15)
             << this->anal_divisions << std::setw(15)
             << this->anal_additions + this->anal_multiplications + this->anal_divisions << std::setw(15)
             << this->anal_intensity << std::endl;
}


void layer_analyzer::calc_intensity() {
#if DO_LAYER_ANALYSIS
    this->anal_intensity = static_cast<float>(this->anal_additions + this->anal_multiplications + this->anal_divisions) / (this->anal_loads + this->anal_stores);
#endif
}

void layer_analyzer::incr_loads() {
#if DO_LAYER_ANALYSIS
    this->anal_loads++;
#endif
}

void layer_analyzer::incr_stores() {
#if DO_LAYER_ANALYSIS
    this->anal_stores++;
#endif
}

void layer_analyzer::incr_additions() {
#if DO_LAYER_ANALYSIS
    this->anal_additions++;
#endif
}

void layer_analyzer::incr_multiplications() {
#if DO_LAYER_ANALYSIS
    this->anal_multiplications++;
#endif
}

void layer_analyzer::incr_divisions() {
#if DO_LAYER_ANALYSIS
    this->anal_divisions++;
#endif
}
