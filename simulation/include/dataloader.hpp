#ifndef DATALOADER_H
#define DATALOADER_H
#include <stdint.h>
#include <vector>
#include <tuple>
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

    uint16_t dataloader_total_size;
    uint16_t dataloader_train_size;
    uint16_t dataloader_validation_size;
    uint8_t  dataloader_batch_size;
    uint16_t dataloader_num_batches;
    std::vector<uint32_t> shuffle_vector(std::vector<uint32_t> vector);
    std::vector<float> shuffle_vector(std::vector<float> vector);
public:
    std::vector<float> layer_outputs;
    
    dataloader(std::vector<std::string> words, std::string speaker_id, uint8_t dataloader_batch_size = 1, float dataloader_train_split = 0.9, std::string dataset_path = "../dataset_mfccs_raw/");
    ~dataloader();


    std::vector<std::string> get_path_names_train();
    std::vector<std::vector<float>> get_inputs_train();
    std::vector<uint8_t> get_labels_train();
    std::vector<uint32_t> get_pick_list();
    std::tuple<std::vector<float>,std::vector<uint8_t>> get_batch();
    std::tuple<std::vector<std::vector<float>>,std::vector<uint8_t>> get_validation_set();
    bool get_training_pool_empty();
    uint16_t get_validation_size();
    void reset_training_pool();
    void print_progress_bar(uint32_t epoch, float error = 0.1);

};


#endif