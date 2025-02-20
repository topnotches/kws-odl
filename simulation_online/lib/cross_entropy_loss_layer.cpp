
#include "cross_entropy_loss_layer.hpp"
#include <math.h>
#include "layer_analyzer.hpp"
extern layer_analyzer crossloss_analyzer;


void cross_entropy_loss_sequential(const float* crossloss_true_labels, const float* crossloss_predicted_labels, float* loss, const uint8_t crossloss_batch_size, const uint8_t crossloss_num_labels) {
    const float small_number = 1e-8;
    
#if DO_LAYER_ANALYSIS
#else
    //#pragma omp parallel for
#endif
    for (uint8_t index_batch = 0; index_batch < crossloss_batch_size; index_batch++) {
        float temp_loss = 0;
        for (uint8_t index_label = 0; index_label < crossloss_num_labels; index_label++) {
            temp_loss += crossloss_true_labels[index_label + index_batch * crossloss_num_labels] * logf(crossloss_predicted_labels[index_label + index_batch * crossloss_num_labels] + small_number);
            crossloss_analyzer.incr_loads();
            crossloss_analyzer.incr_loads();
            crossloss_analyzer.incr_loads();
            crossloss_analyzer.incr_additions();
            crossloss_analyzer.incr_multiplications();
        }
        loss[index_batch] = -temp_loss;
        
        crossloss_analyzer.incr_stores();
        crossloss_analyzer.incr_additions();
        
    }
}