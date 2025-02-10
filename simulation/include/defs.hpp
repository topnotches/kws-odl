#ifndef DEFS_H
#define DEFS_H
#include <stdio.h>
#include "layer_analyzer.hpp"

#define DO_LAYER_ANALYSIS false // change batch-size to 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#define DO_FULL_BATCHNORM_LAYER_ANALYSIS false 




#define NUMBER_OF_CLASSES   12
#define EPOCHS              1000
#define TOTAL_RUNS          1000
#define BATCH_SIZE          1
#define LAYER_SELECT        28


#define LENGTH_BIT          1
#define LENGTH_NIBBLE       4
#define LENGTH_BYTE         8
#define LENGTH_HALFWORD     16
#define LENGTH_WORD         32
#define LENGTH_DOUBLE_WORD  64

enum class LayerTypes {
    conv,
    dw_conv,
    dense,
    batchnorm,
    relu,
    avgpool2d,
    softmax,
    cross_entropy_loss,
    fusion
};

typedef struct conv_hypr_param_t {
    uint8_t pad_top = 0;
    uint8_t pad_bottom = 0;
    uint8_t pad_left = 0;
    uint8_t pad_right = 0;

    uint8_t kernel_stride = 0;
    uint8_t kernel_width = 0;
    uint8_t kernel_height = 0;
    uint8_t kernel_count = 0;
} conv_hypr_param;


typedef struct dense_hypr_param_t {
    uint32_t size_in = 0;
    uint32_t size_out = 0;
} dense_param;

typedef struct tensor_dim_sizes_t {
    uint32_t full = 0;
    uint16_t width = 0;
    uint16_t height = 0;
    uint16_t depth = 0;
    uint16_t batch = 0;
} tensor_dim_sizes;

#endif
