#ifndef DEFS_H
#define DEFS_H
#include <stdio.h>
#include "layer_analyzer.hpp"

#define DO_LAYER_ANALYSIS false // change batch-size to 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#define DO_FULL_BATCHNORM_LAYER_ANALYSIS false 



extern volatile double DENSE_BW_OUTPUT_SCALE;

/***************************/
/* MODEL FORMAT PARAMETERS */
/***************************/
// Requantization select and compensation order
#define USE_SHIFT_REQUANT                 true
#define DO_THIRD_ORDER_SHIFT_REQUANT      true 

typedef struct quant_param_t {
    double scale_in = 1;
    double scale_out = 1;
    double scale_weight = 1;
    uint8_t weight_bits = 1;
    uint8_t activation_bits = 1;
    uint8_t gradient_bits = 1;
} quant_param;

// Layer 10, Fusion
#define LAYER_10_QPARAM_WEIGHT_BITS 21
#define LAYER_10_QPARAM_ACTIVA_BITS 8
#define LAYER_10_QPARAM_GRADNT_BITS LAYER_10_QPARAM_WEIGHT_BITS
#define FUSION_QPARAM_WEIGHT_SCALE_SHIFT LAYER_10_QPARAM_WEIGHT_BITS-2 // FIXME & TODO REAL NUMBER BRR

// QUANTIZATION PARAMETERS FOR THE REST OF THE LAYERS:

// Layer 1, Convolution
#define LAYER_1_QPARAM_WEIGHT_BITS 8
#define LAYER_1_QPARAM_ACTIVA_BITS 8
#define LAYER_1_QPARAM_GRADNT_BITS 1

// Layer 2, Depthwise Convolution
#define LAYER_2_QPARAM_WEIGHT_BITS 8
#define LAYER_2_QPARAM_ACTIVA_BITS 8
#define LAYER_2_QPARAM_GRADNT_BITS 1

// Layer 3, Pointwise Convolution
#define LAYER_3_QPARAM_WEIGHT_BITS 8
#define LAYER_3_QPARAM_ACTIVA_BITS 8
#define LAYER_3_QPARAM_GRADNT_BITS 1

// Layer 4, Depthwise Convolution
#define LAYER_4_QPARAM_WEIGHT_BITS 8
#define LAYER_4_QPARAM_ACTIVA_BITS 8
#define LAYER_4_QPARAM_GRADNT_BITS 1

// Layer 5, Pointwise Convolution
#define LAYER_5_QPARAM_WEIGHT_BITS 8
#define LAYER_5_QPARAM_ACTIVA_BITS 8
#define LAYER_5_QPARAM_GRADNT_BITS 1

// Layer 6, Depthwise Convolution
#define LAYER_6_QPARAM_WEIGHT_BITS 8
#define LAYER_6_QPARAM_ACTIVA_BITS 8
#define LAYER_6_QPARAM_GRADNT_BITS 1

// Layer 7, Pointwise Convolution
#define LAYER_7_QPARAM_WEIGHT_BITS 8 
#define LAYER_7_QPARAM_ACTIVA_BITS 8
#define LAYER_7_QPARAM_GRADNT_BITS 1

// Layer 8, Depthwise Convolution
#define LAYER_8_QPARAM_WEIGHT_BITS 8
#define LAYER_8_QPARAM_ACTIVA_BITS 8
#define LAYER_8_QPARAM_GRADNT_BITS 1

// Layer 9, Pointwise Convolution
#define LAYER_9_QPARAM_WEIGHT_BITS 8
#define LAYER_9_QPARAM_ACTIVA_BITS 8
#define LAYER_9_QPARAM_GRADNT_BITS 1

// Layer 11, Dense
#define LAYER_11_QPARAM_WEIGHT_BITS 8
#define LAYER_11_QPARAM_ACTIVA_BITS 8
#define LAYER_11_QPARAM_GRADNT_BITS LAYER_10_QPARAM_WEIGHT_BITS

// Layer 12, SoftMax
#define LAYER_SOFTMAX_QPARAM_WEIGHT_BITS 8
#define LAYER_SOFTMAX_QPARAM_ACTIVA_BITS 8
#define LAYER_SOFTMAX_QPARAM_GRADNT_BITS LAYER_11_QPARAM_WEIGHT_BITS

/*****************************/
/* SIMULATION RUN PARAMETERS */
/*****************************/
#define NUMBER_OF_CLASSES   12
#define EPOCHS              20000
#define TOTAL_RUNS          1
#define TRAIN_VAL_SPLIT     0.7
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
