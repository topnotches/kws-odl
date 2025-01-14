#ifndef DEFS_H
#define DEFS_H
#include <stdio.h>
#include "misc_utils.hpp"


enum class LayerTypes {
    conv,
    dw_conv,
    dense,
    batchnorm,
    relu
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

typedef struct dense_param_t {
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