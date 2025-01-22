
#include "softmax_layer.hpp"
#include <math.h>


void cross_entropy_loss_sequential(const float* crossloss_true_labels, const float* crossloss_predicted_labels, float* loss, const uint8_t crossloss_batch_size, const uint8_t crossloss_num_labels) {
    const float small_number = 1e-8;
    
    for (uint8_t index_batch = 0; index_batch < crossloss_batch_size; index_batch++) {
        float temp_loss = 0;
        //#pragma omp parallel for
        for (uint8_t index_label = 0; index_label < crossloss_num_labels; index_label++) {
            temp_loss += crossloss_true_labels[index_label + index_batch * crossloss_num_labels] * logf(crossloss_predicted_labels[index_label + index_batch * crossloss_num_labels] + small_number);
        }
        loss[index_batch] = -temp_loss;
    }
}