
#include "dataloader.hpp"
#include "misc_utils.hpp"
#include <string>
#include <iostream>
#include <algorithm>
#include <iomanip>

#include <filesystem>
#include <string>
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

dataloader::dataloader(std::vector<std::string> dataloader_target_words, std::string dataloader_speaker_id, uint8_t dataloader_batch_size, float dataloader_train_split, std::string dataloader_dataset_path, bool do_shuffle) {

    this->do_shuffle = do_shuffle;
    uint8_t label = 0;
    this->dataloader_path_names_train = {};
    this->dataloader_inputs_train_float = {};
    this->dataloader_inputs_train_fixed = {};
    this->dataloader_labels_train = {};
    this->dataloader_path_names_validation = {};
    this->dataloader_inputs_validation_float = {};
    this->dataloader_inputs_validation_fixed = {};
    this->dataloader_labels_validation = {};
    this->dataloader_pick_list = {};
    
    std::vector<std::string>        temp_dataloader_path_names;
    std::vector<std::vector<float>> temp_dataloader_inputs_float;
    std::vector<std::vector<int32_t>> temp_dataloader_inputs_fixed;
    std::vector<uint8_t>            temp_dataloader_labels;

    for (std::string target_word : dataloader_target_words) {
        std::string dir_path = dataloader_dataset_path + target_word;
        for (const auto & entry : fs::directory_iterator(dir_path)) {
            if (entry.path().string().find(dataloader_speaker_id) != std::string::npos) {
            //    std::cout << entry.path().string() << std::endl;
                temp_dataloader_path_names.push_back(entry.path().string());
                temp_dataloader_inputs_float.push_back(load_mffcs_bin_float(entry.path().string()));
                temp_dataloader_inputs_fixed.push_back(load_mffcs_bin_fixed(entry.path().string()));
                temp_dataloader_labels.push_back(label);
            }
        }
        label++;
    }
    
    std::vector<uint32_t> indexes;
    for (uint32_t i = 0; i < temp_dataloader_path_names.size(); ++i) {
        indexes.push_back(i);
    }

    if (this->do_shuffle) {
        indexes = this->shuffle_vector(indexes);
    }
    this->dataloader_total_size = indexes.size();
    this->dataloader_train_size = (uint32_t) ((float) dataloader_total_size * dataloader_train_split);
    this->dataloader_validation_size = dataloader_total_size - dataloader_train_size;
    this->dataloader_batch_size = dataloader_batch_size;
    this->dataloader_num_batches = this->dataloader_train_size / this->dataloader_batch_size; 
    printf("Splitting dataset into train/validation: \n");
    printf("Total size:      %d \n",this->dataloader_total_size);
    printf("Train size:      %d \n",this->dataloader_train_size);
    printf("Validation size: %d \n",this->dataloader_validation_size);
    printf("Batch size:      %d \n",this->dataloader_batch_size);
    
    for (uint32_t i = 0; i < indexes.size(); i++) {
        if (i < this->dataloader_train_size) {
            this->dataloader_path_names_train.push_back(temp_dataloader_path_names[indexes[i]]);
            this->dataloader_inputs_train_float.push_back(temp_dataloader_inputs_float[indexes[i]]);
            this->dataloader_inputs_train_fixed.push_back(temp_dataloader_inputs_fixed[indexes[i]]);
            this->dataloader_labels_train.push_back(temp_dataloader_labels[indexes[i]]);
        } else {
            this->dataloader_path_names_validation.push_back(temp_dataloader_path_names[indexes[i]]);
            this->dataloader_inputs_validation_float.push_back(temp_dataloader_inputs_float[indexes[i]]);
            this->dataloader_inputs_validation_fixed.push_back(temp_dataloader_inputs_fixed[indexes[i]]);
            this->dataloader_labels_validation.push_back(temp_dataloader_labels[indexes[i]]);
        }
    }
    // Initialize pick list
    for (uint32_t i = 0; i < this->dataloader_train_size; ++i) {
        this->dataloader_pick_list.push_back(i);
    }
    if (this->do_shuffle) {
        this->dataloader_pick_list = this->shuffle_vector(this->dataloader_pick_list);
    }
    //for (uint32_t i = 0; i < this->dataloader_train_size; ++i) {
    //    
    //    std::cout << "picklist index: " << this->dataloader_pick_list[i] << " " << "picklist item: " << this->dataloader_path_names_train[this->dataloader_pick_list[i]] << " " << "index: " << i << " ";
    //    std::cout << this->dataloader_path_names_train[i] << std::endl;
    //}
}

dataloader::~dataloader( ) {
    
}

std::tuple<std::vector<float>,std::vector<uint8_t>> dataloader::get_batch_float() {
    std::vector<float> input_data;
    std::vector<uint8_t> input_labels;
    if (this->dataloader_batch_size <= this->dataloader_pick_list.size()) {
        for (uint8_t i = 0; i < this->dataloader_batch_size; i++) {
            //std::cout << "Picked name: " <<  this->dataloader_path_names_train[this->dataloader_pick_list.back()]<<std::endl;

            std::vector<float> temp_input_data = this->dataloader_inputs_train_float[this->dataloader_pick_list.back()];
            input_data.insert(input_data.end(), temp_input_data.begin(), temp_input_data.end());
            
            uint8_t temp_input_labels = this->dataloader_labels_train[this->dataloader_pick_list.back()];
            this->dataloader_pick_list.pop_back();
            input_labels.push_back(temp_input_labels);
        }
    } else {
        KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("ERROR: batch size is larger than residuals");
    }
    return std::make_tuple(input_data, input_labels);
}


std::tuple<std::vector<int32_t>,std::vector<uint8_t>> dataloader::get_batch_fixed() {
    std::vector<int32_t> input_data;
    std::vector<uint8_t> input_labels;
    if (this->dataloader_batch_size <= this->dataloader_pick_list.size()) {
        for (uint8_t i = 0; i < this->dataloader_batch_size; i++) {
            //std::cout << "Picked name: " <<  this->dataloader_path_names_train[this->dataloader_pick_list.back()]<<std::endl;

            std::vector<int32_t> temp_input_data = this->dataloader_inputs_train_fixed[this->dataloader_pick_list.back()];
            input_data.insert(input_data.end(), temp_input_data.begin(), temp_input_data.end());
            
            uint8_t temp_input_labels = this->dataloader_labels_train[this->dataloader_pick_list.back()];
            this->dataloader_pick_list.pop_back();
            input_labels.push_back(temp_input_labels);
        }
    } else {
        KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("ERROR: batch size is larger than residuals");
    }
    return std::make_tuple(input_data, input_labels);
}


std::tuple<std::tuple<std::vector<int32_t>,std::vector<uint8_t>>, std::tuple<std::vector<float>,std::vector<uint8_t>>> dataloader::get_batch_fixed_and_float() {
    std::vector<float> input_data_float;
    std::vector<int32_t> input_data;
    std::vector<uint8_t> input_labels;
    if (this->dataloader_batch_size <= this->dataloader_pick_list.size()) {
        for (uint8_t i = 0; i < this->dataloader_batch_size; i++) {
            //std::cout << "Picked name: " <<  this->dataloader_path_names_train[this->dataloader_pick_list.back()]<<std::endl;
            std::vector<float> temp_input_data_float = this->dataloader_inputs_train_float[this->dataloader_pick_list.back()];
            input_data_float.insert(input_data_float.end(), temp_input_data_float.begin(), temp_input_data_float.end());
            
            std::vector<int32_t> temp_input_data = this->dataloader_inputs_train_fixed[this->dataloader_pick_list.back()];
            input_data.insert(input_data.end(), temp_input_data.begin(), temp_input_data.end());
            
            uint8_t temp_input_labels = this->dataloader_labels_train[this->dataloader_pick_list.back()];
            this->dataloader_pick_list.pop_back();
            input_labels.push_back(temp_input_labels);
        }
    } else {
        KHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAN("ERROR: batch size is larger than residuals");
    }
    return std::make_tuple(std::make_tuple(input_data, input_labels), std::make_tuple(input_data_float, input_labels));
}

std::tuple<std::vector<std::vector<float>>,std::vector<uint8_t>> dataloader::get_validation_set_float() {
    std::vector<std::vector<float>> input_data;
    std::vector<uint8_t> input_labels;
    for (auto val_data : this->dataloader_inputs_validation_float) {
            input_data.push_back(val_data);
    }
    for (auto val_label : this->dataloader_labels_validation) {
        input_labels.push_back(val_label);
    }
    return std::make_tuple(input_data, input_labels);
}
std::tuple<std::vector<std::vector<int32_t>>,std::vector<uint8_t>> dataloader::get_validation_set_fixed() {
    std::vector<std::vector<int32_t>> input_data;
    std::vector<uint8_t> input_labels;
    for (auto val_data : this->dataloader_inputs_validation_fixed) {
            input_data.push_back(val_data);
    }
    for (auto val_label : this->dataloader_labels_validation) {
        input_labels.push_back(val_label);
    }
    return std::make_tuple(input_data, input_labels);
}

void dataloader::print_progress_bar(uint32_t epoch, float error) {
    float progress = (float)(dataloader_num_batches - (this->dataloader_pick_list.size() / this->dataloader_batch_size)) / dataloader_num_batches;

    int barWidth = 70;

    std::cout << "\x1b[1A" // Move cursor up one
        << "\x1b[2K"; // Delete the entire line
    std::cout << "EPOCH: " << epoch << "/" << EPOCHS << "[";
    int pos = barWidth * progress;
    
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << std::setfill(' ') << int(progress * 100) << "%, Loss: " << error << "\r";
    std::cout.flush();

    std::cout << std::endl;
}

uint16_t dataloader::get_validation_size() {
    return this->dataloader_validation_size;
}
bool dataloader::get_training_pool_empty() {
    if (this->dataloader_batch_size <= this->dataloader_pick_list.size()) {
        return 1;
    } else {
        return 0;
    }
}

void dataloader::reset_training_pool() {
    // Initialize pick list
    for (uint32_t i = 0; i < this->dataloader_train_size; ++i) {
        this->dataloader_pick_list.push_back(i);
    }
    if (this->do_shuffle) {
        this->dataloader_pick_list = this->shuffle_vector(this->dataloader_pick_list);
    }
}

std::vector<std::string> dataloader::get_path_names_train() {
    return this->dataloader_path_names_train;
}

std::vector<std::vector<float>> dataloader::get_inputs_train_float() {
    return this->dataloader_inputs_train_float;
}
std::vector<std::vector<int32_t>> dataloader::get_inputs_train_fixed() {
    return this->dataloader_inputs_train_fixed;
}

std::vector<uint8_t> dataloader::get_labels_train() {
    return this->dataloader_labels_train;
}

std::vector<uint32_t> dataloader::get_pick_list() {
    return this->dataloader_pick_list;
}

std::vector<uint32_t> dataloader::shuffle_vector(std::vector<uint32_t> vector) {
    std::srand(time(NULL));
    std::random_shuffle(vector.begin(), vector.end());
    return vector;
}
std::vector<int32_t> dataloader::shuffle_vector(std::vector<int32_t> vector) {
    std::srand(time(NULL));
    std::random_shuffle(vector.begin(), vector.end());
    return vector;
}
std::vector<float> dataloader::shuffle_vector(std::vector<float> vector) {
    std::srand(time(NULL));
    std::random_shuffle(vector.begin(), vector.end());
    return vector;
}

void dataloader::set_train_set(std::vector<std::vector<float>> cached_list) {
    this->dataloader_inputs_train_float = cached_list;
}
void dataloader::set_train_set(std::vector<std::vector<int32_t>> cached_list) {
    this->dataloader_inputs_train_fixed = cached_list;
}

void dataloader::set_validation_set(std::vector<std::vector<float>> cached_list) {
    this->dataloader_inputs_validation_float = cached_list;
}
void dataloader::set_validation_set(std::vector<std::vector<int32_t>> cached_list) {
    this->dataloader_inputs_validation_fixed = cached_list;
}