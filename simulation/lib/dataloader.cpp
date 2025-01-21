
#include "dataloader.hpp"
#include "misc_utils.hpp"
#include <string>
#include <iostream>
#include <algorithm>

#include <filesystem>
#include <string>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

dataloader::dataloader(std::vector<std::string> dataloader_target_words, std::string dataloader_speaker_id, std::string dataloader_dataset_path, float train_split) {


    uint8_t label = 0;
    this->dataloader_path_names_train = {};
    this->dataloader_inputs_train = {};
    this->dataloader_labels_train = {};
    this->dataloader_path_names_validation = {};
    this->dataloader_inputs_validation = {};
    this->dataloader_labels_validation = {};
    this->dataloader_pick_list = {};
    
    std::vector<std::string>        temp_dataloader_path_names;
    std::vector<std::vector<float>> temp_dataloader_inputs;
    std::vector<uint8_t>            temp_dataloader_labels;

    for (std::string target_word : dataloader_target_words) {
        std::string dir_path = dataloader_dataset_path + target_word;
        for (const auto & entry : fs::directory_iterator(dir_path)) {
            if (entry.path().string().find(dataloader_speaker_id) != std::string::npos) {
                std::cout << entry.path().string() << std::endl;
                temp_dataloader_path_names.push_back(entry.path().string());
                temp_dataloader_inputs.push_back(load_mffcs_bin(entry.path().string()));
                temp_dataloader_labels.push_back(label);
            }
        }
        label++;
    }
    
    std::vector<uint32_t> indexes;
    for (uint32_t i = 0; i < temp_dataloader_path_names.size(); ++i)
        indexes.push_back(i);
    std::random_shuffle(indexes.begin(), indexes.end());
    uint32_t total = indexes.size();
    uint32_t train = (uint32_t) ((float) total * train_split);
    uint32_t validation = total - train;
    
    printf("Splitting dataset into train/validation: \n");
    printf("Total      %d: \n",total);
    printf("Train      %d: \n",train);
    printf("Validation %d: \n",validation);
    
    for (uint32_t i = 0; i < indexes.size(); i++) {
        
    }

    for (uint8_t i = 0; i < this->dataloader_labels_train.size(); i++) {
        printf("index: %d \n", i);
        printf(this->dataloader_path_names_train[i].c_str());
        printf("\ndataloader_label: %d \n",this->dataloader_labels_train[i]);
    }

}
dataloader::~dataloader( ) {
    
}

std::vector<std::string> dataloader::get_dataloader_path_names_train() {
    return this->dataloader_path_names_train;
}

std::vector<std::vector<float>> dataloader::get_dataloader_inputs_train() {
    return this->dataloader_inputs_train;
}

std::vector<uint8_t> dataloader::get_dataloader_labels_train() {
    return this->dataloader_labels_train;
}

std::vector<uint32_t> dataloader::get_dataloader_pick_list() {
    return this->dataloader_pick_list;
}