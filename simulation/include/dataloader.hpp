#ifndef DATALOADER_H
#define DATALOADER_H
#include <stdint.h>
#include <vector>
#include <string>
#include "defs.hpp"
class dataloader {
private:
    std::vector<std::string>        dataloader_path_names_train;
    std::vector<std::vector<float>> dataloader_inputs_train;
    std::vector<uint8_t>            dataloader_labels_train;
    std::vector<std::string>        dataloader_path_names_validation;
    std::vector<std::vector<float>> dataloader_inputs_validation;
    std::vector<uint8_t>            dataloader_labels_validation;
    
    std::vector<uint32_t>           dataloader_pick_list;
public:
    std::vector<float> layer_outputs;
    
    dataloader(std::vector<std::string> words, std::string speaker_id, std::string dataset_path = "../dataset_mfccs_raw/", float train_split = 0.9);
    ~dataloader();
    std::vector<std::string> get_dataloader_path_names_train();
    std::vector<std::vector<float>> get_dataloader_inputs_train();
    std::vector<uint8_t> get_dataloader_labels_train();
    std::vector<uint32_t> get_dataloader_pick_list();

};


#endif