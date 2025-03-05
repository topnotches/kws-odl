# Copyright (C) 2021 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
#
# Author: Cristian Cioflan, ETH (cioflanc@iis.ee.ethz.ch)


import os

from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.ao.quantization as quant

import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
def get_qconfig8(bits_w,bits_a):
    qmax = 2**bits_a - 1
    qmin_signed = -2**(bits_w-1)
    qmax_signed = 2**(bits_w-1) - 1

    return quant.QConfig(
        activation=quant.FakeQuantize.with_args(
            observer=quant.MovingAverageMinMaxObserver,
            quant_min=0, quant_max=qmax,
            dtype=torch.qint32,
            qscheme=torch.per_tensor_affine
        ),
        weight=quant.FakeQuantize.with_args(
            observer=quant.MovingAverageMinMaxObserver,
            quant_min=qmin_signed, quant_max=qmax_signed,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )
    )
def get_qconfig8_sign(bits_w,bits_a):
    qmina_signed = -2**(bits_a-1)
    qmaxa_signed = 2**(bits_a-1) - 1
    qminw_signed = -2**(bits_w-1)
    qmaxw_signed = 2**(bits_w-1) - 1

    return quant.QConfig(
        activation=quant.FakeQuantize.with_args(
            observer=quant.MovingAverageMinMaxObserver,
            quant_min=qmina_signed, quant_max=qmaxa_signed,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        ),
        weight=quant.FakeQuantize.with_args(
            observer=quant.MovingAverageMinMaxObserver,
            quant_min=qminw_signed, quant_max=qmaxw_signed,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )
    )
def get_qconfig16(bits):
    qmax = 2**bits - 1
    qmin_signed = -2**(bits-1)
    qmax_signed = 2**(bits-1) - 1

    return quant.QConfig(
        activation=quant.FakeQuantize.with_args(
            observer=quant.MovingAverageMinMaxObserver,
            quant_min=0, quant_max=qmax,
            dtype=torch.qint32,
            qscheme=torch.per_tensor_affine
        ),
        weight=quant.FakeQuantize.with_args(
            observer=quant.MovingAverageMinMaxObserver,
            quant_min=qmin_signed, quant_max=qmax_signed,
            dtype=torch.qint32,
            qscheme=torch.per_tensor_symmetric
        )
    )

def npy_to_txt(layer_number, activations):
    # Saving the input

    if layer_number == -1:
        tmp = activations.reshape(-1)
        f = open('input.txt', "a")
        f.write('# input (shape [1, 49, 10]),\\\n')
        for elem in tmp:
            if (elem < 0):
                f.write (str(256+elem) + ",\\\n")
            else:
                f.write (str(elem) + ",\\\n")
        f.close()
    # Saving layers' activations
    else:
        tmp = activations.reshape(-1)
        f = open('out_layer' + str(layer_number) + '.txt', "a")
        f.write('layers.0.relu1 (shape [1, 25, 5, 64]),\\\n')  # Hardcoded, should be adapted for better understanding.
        for elem in tmp:
            if (elem < 0):
                f.write (str(256+elem) + ",\\\n")
            else:
                f.write (str(elem) + ",\\\n")
        f.close()


def remove_txt():
    # Removing old activations and inputs

    directory = '.'
    files_in_directory = os.listdir(directory)
    filtered_files = [file for file in files_in_directory if (file.startswith("out_layer") or file.startswith("input.txt"))]
    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)


def conf_matrix(labels, predicted, training_parameters):
    # Plotting confusion matrix

    labels = labels.cpu()
    predicted = predicted.cpu()
    cm = confusion_matrix(labels, predicted)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, index = [i for i in ['silence','unknown']+training_parameters['wanted_words']],
                  columns = [i for i in ['silence','unknown']+training_parameters['wanted_words']])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


def parameter_generation():
    # Data processing parameters

    data_processing_parameters = {
    'feature_bin_count':10
    }
    time_shift_ms=200
    sample_rate=16000
    clip_duration_ms=1000
    time_shift_samples= int((time_shift_ms * sample_rate) / 1000)
    window_size_ms=40.0
    window_stride_ms=20.0
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    data_processing_parameters['desired_samples'] = desired_samples
    data_processing_parameters['sample_rate'] = sample_rate
    data_processing_parameters['spectrogram_length'] = spectrogram_length
    data_processing_parameters['window_stride_samples'] = window_stride_samples
    data_processing_parameters['window_size_samples'] = window_size_samples

    # Training parameters
    training_parameters = {
    'data_dir':'../dataset',
    'data_url':'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
    'epochs':40,
    'batch_size':4,
    'silence_percentage':0.0,
    'unknown_percentage':0.0,
    'validation_percentage':10.0,
    'testing_percentage':0.0,
    'background_frequency':0.8,
    'background_volume':0.2,
    }
    target_words='yes,no,up,down,left,right,on,off,stop,go,'  # GSCv2 - 12 words
    # Selecting 35 words
    # target_words='yes,no,up,down,left,right,on,off,stop,go,backward,bed,bird,cat,dog,eight,five,follow,forward,four,happy,house,learn,marvin,nine,one,seven,sheila,six,three,tree,two,visual,wow,zero,'  # GSCv2 - 35 words
    wanted_words=(target_words).split(',')
    wanted_words.pop()
    training_parameters['wanted_words'] = wanted_words
    training_parameters['time_shift_samples'] = time_shift_samples

    return training_parameters, data_processing_parameters

quant_fuse_list = [
            ["ConvBNReLU1.0","ConvBNReLU1.1","ConvBNReLU1.2"],
            ["ConvBNReLU2.0","ConvBNReLU2.1","ConvBNReLU2.2"],
            ["ConvBNReLU3.0","ConvBNReLU3.1","ConvBNReLU3.2"],
            ["ConvBNReLU4.0","ConvBNReLU4.1","ConvBNReLU4.2"],
            ["ConvBNReLU5.0","ConvBNReLU5.1","ConvBNReLU5.2"],
            ["ConvBNReLU6.0","ConvBNReLU6.1","ConvBNReLU6.2"],
            ["ConvBNReLU7.0","ConvBNReLU7.1","ConvBNReLU7.2"],
            ["ConvBNReLU8.0","ConvBNReLU8.1","ConvBNReLU8.2"],
            ["ConvBNReLU9.0","ConvBNReLU9.1","ConvBNReLU9.2"],
        ]
qat_configs = {
    "ConvBNReLU1.0": get_qconfig8(8,8),
    "ConvBNReLU2.0": get_qconfig8(8,8),
    "ConvBNReLU3.0": get_qconfig8(8,8),
    "ConvBNReLU4.0": get_qconfig8(8,8),
    "ConvBNReLU5.0": get_qconfig8(8,8),
    "ConvBNReLU6.0": get_qconfig8(8,8),
    "ConvBNReLU7.0": get_qconfig8(8,8),
    "ConvBNReLU8.0": get_qconfig8(8,8),
    "ConvBNReLU9.0": get_qconfig8(8,8),
    "fc1": get_qconfig8_sign(8,8),
}

STEP_DO_QAT_TRAIN               = False
STEP_DO_TRAIN                   = False
STEP_DO_EXPORT_MODEL_FLOAT      = False
STEP_DO_EXPORT_MODEL_FIXED      = True
STEP_DO_PROCESS_MFCCS_FLOAT     = False
STEP_DO_PROCESS_MFCCS_FIXED     = False # not implemented
CHECKPOINT_PATH_FLOAT           = 'none'
CHECKPOINT_PATH_FIXED           = './run_qat_88_48_48_48_88_88_bs_4_ACTUAL_split_3/model_acc_91.40625_03_03_2025_184950.pth'
CLASSES                         = 10
NOT_PRETRAIN_BUT_OL_WORD_LIMIT  = 3
HYPERPARAMETER_SETUP            = 'qat_88_88_88_88_88_88_bs_8_ACTUAL_split_3'
CHECKPOINT_SAVE_PATH            = './run_' + HYPERPARAMETER_SETUP
EXPORT_OUTPUT_DIR_PATH_FLOAT    = '../simulation_online/'
EXPORT_OUTPUT_DIR_PATH_FIXED    = '../simulation_online/'
EXPORT_OUTPUT_NAME_FLOAT        = 'export_params_nclass_' + str(CLASSES) + '.csv'
EXPORT_OUTPUT_NAME_FIXED        = 'qat_export_params_nclass_' + str(CLASSES) + '.csv'
EXPORT_OUTPUT_PATH_FLOAT        = EXPORT_OUTPUT_DIR_PATH_FLOAT + EXPORT_OUTPUT_NAME_FLOAT
EXPORT_OUTPUT_PATH_FIXED        = EXPORT_OUTPUT_DIR_PATH_FIXED + EXPORT_OUTPUT_NAME_FIXED
MFCCS_INPUT_PATHS               = ['../dataset_mfccs_raw/yes/d21fd169_nohash_0',
                                    '../dataset_mfccs_raw/yes/d21fd169_nohash_1']  # Path(s) to MFCCs binary file
MFCCS_OUTPUT_PATH               = './output_mfccs.bin' # Path to save model output
