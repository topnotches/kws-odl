#ifndef CROSS_ENTROPY_LOSS_LAYER_H
#define CROSS_ENTROPY_LOSS_LAYER_H

#include <stdint.h>

void cross_entropy_loss_float(const float* crossloss_true_labels, const float* crossloss_predicted_labels, float* loss, const uint8_t crossloss_batch_size, const uint8_t crossloss_num_labels);

void cross_entropy_loss_fixed(const int32_t* crossloss_true_labels, const int32_t* crossloss_predicted_labels, int32_t* loss, const uint8_t crossloss_batch_size, const uint8_t crossloss_num_labels);

#endif
