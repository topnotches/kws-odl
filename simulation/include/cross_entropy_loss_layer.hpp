#ifndef CROSS_ENTROPY_LOSS_LAYER_H
#define CROSS_ENTROPY_LOSS_LAYER_H

#include <stdint.h>

void cross_entropy_loss_sequential(const float* crossloss_true_labels, const float* crossloss_predicted_labels, float* loss, const uint8_t crossloss_batch_size, const uint8_t crossloss_num_labels);

#endif
