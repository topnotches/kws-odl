#!/usr/bin/env python3


import hashlib
import math
import os.path
import random
import shutil
import os
import logging
import re
import glob
import time
import torch
import torchaudio

from collections import Counter, OrderedDict

import soundfile as sf
import numpy as np
import tensorflow as tf

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
BACKGROUND_NOISE_LABEL = "_background_noise_"
SILENCE_LABEL = "_silence_"
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = "_unknown_"
UNKNOWN_WORD_INDEX = 1
RANDOM_SEED = 59185

DO_SHIFT_JESPER = 1
DO_NOISE_JESPER = 1

# Data processing parameters

data_processing_parameters = {"feature_bin_count": 10}
time_shift_ms = 200
sample_rate = 16000
clip_duration_ms = 1000
time_shift_samples = int((time_shift_ms * sample_rate) / 1000)
window_size_ms = 40.0
window_stride_ms = 20.0
desired_samples = int(sample_rate * clip_duration_ms / 1000)
window_size_samples = int(sample_rate * window_size_ms / 1000)
window_stride_samples = int(sample_rate * window_stride_ms / 1000)
length_minus_window = desired_samples - window_size_samples
if length_minus_window < 0:
    spectrogram_length = 0
else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
data_processing_parameters["desired_samples"] = desired_samples
data_processing_parameters["sample_rate"] = sample_rate
data_processing_parameters["spectrogram_length"] = spectrogram_length
data_processing_parameters["window_stride_samples"] = window_stride_samples
data_processing_parameters["window_size_samples"] = window_size_samples

# Training parameters
training_parameters = {
    "data_dir": "../dataset",
    "data_url": "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
    "epochs": 40,
    "batch_size": 128,
    "silence_percentage": 10.0,
    "unknown_percentage": 10.0,
    "validation_percentage": 10.0,
    "testing_percentage": 10.0,
    "background_frequency": 0.8,
    "background_volume": 0.2,
}
target_words = "yes,no,up,down,left,right,on,off,stop,go,"  # GSCv2 - 12 words
# Selecting 35 words
# target_words='yes,no,up,down,left,right,on,off,stop,go,backward,bed,bird,cat,dog,eight,five,follow,forward,four,happy,house,learn,marvin,nine,one,seven,sheila,six,three,tree,two,visual,wow,zero,'  # GSCv2 - 35 words
wanted_words = (target_words).split(",")
wanted_words.pop()
training_parameters["wanted_words"] = wanted_words
training_parameters["time_shift_samples"] = time_shift_samples


def prepare_words_list(wanted_words):
    return wanted_words


def which_set(filename, validation_percentage, testing_percentage):
    # Split dataset in training, validation, and testing set
    # Should be modified to load validation data from validation_list.txt
    # Should be modified to load testing data from testing_list.txt

    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r"_nohash_.*$", "", base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
    percentage_hash = (int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (
        100.0 / MAX_NUM_WAVS_PER_CLASS
    )
    if percentage_hash < validation_percentage:
        result = "validation"
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = "testing"
    else:
        result = "training"
    return result


class AudioProcessor(object):

    def __init__(self, training_parameters, data_processing_params):

        self.data_directory = training_parameters["data_dir"]
        self.generate_background_noise()
        self.data_processing_parameters = data_processing_params

    def generate_background_noise(self):
        # Load background noise, used to augment clean speech

        self.background_noise = []
        background_dir = os.path.join(self.data_directory, BACKGROUND_NOISE_LABEL)
        if not os.path.exists(background_dir):
            return self.background_noise

        search_path = os.path.join(self.data_directory, BACKGROUND_NOISE_LABEL, "*.wav")
        for wav_path in glob.glob(search_path):
            # List of tensor, each one is a background noise
            sf_loader, _ = sf.read(wav_path)
            wav_file = torch.Tensor(np.array([sf_loader]))
            self.background_noise.append(wav_file[0])

        if not self.background_noise:
            raise Exception("No background wav files were found in " + search_path)

    def get_size(self, mode):
        # Compute data set size

        return len(self.data_set[mode])

    def compute_mfccs_sample(self, some_path="./dataset/down/1970b130_nohash_2.wav"):

        # Define sample
        word = some_path.split("/")[-2]
        wav_path = some_path
        speaker_id = some_path.split("/")[-1].split("_")[0]
        sample = {"label": word, "file": wav_path, "speaker": speaker_id}

        # FIXME AND TODO: Use for training on the hearing aid???????
        use_background = self.background_noise

        # Compute time shift offset
        if training_parameters["time_shift_samples"] > 0:
            time_shift_amount = np.random.randint(
                -training_parameters["time_shift_samples"],
                training_parameters["time_shift_samples"],
            )
        else:
            time_shift_amount = 0

        # FIXME AND TODO: Should I timeshift on hearing aid???? ??????????
        if DO_SHIFT_JESPER:
            time_shift_amount = 0

        if time_shift_amount > 0:
            time_shift_padding = [[time_shift_amount, 0], [0, 0]]
            time_shift_offset = [0, 0]
        else:
            time_shift_padding = [[0, -time_shift_amount], [0, 0]]
            time_shift_offset = [-time_shift_amount, 0]

        data_augmentation_parameters = {
            "wav_filename": sample["file"],
            "time_shift_padding": time_shift_padding,
            "time_shift_offset": time_shift_offset,
        }

        # Select background noise to mix in.
        if (use_background or sample["label"] == SILENCE_LABEL) and DO_NOISE_JESPER:
            background_index = np.random.randint(len(self.background_noise))
            background_samples = self.background_noise[background_index].numpy()
            assert (
                len(background_samples)
                > self.data_processing_parameters["desired_samples"]
            )

            background_offset = np.random.randint(
                0,
                len(background_samples)
                - self.data_processing_parameters["desired_samples"],
            )
            background_clipped = background_samples[
                background_offset : (
                    background_offset
                    + self.data_processing_parameters["desired_samples"]
                )
            ]
            background_reshaped = background_clipped.reshape(
                [self.data_processing_parameters["desired_samples"], 1]
            )

            if sample["label"] == SILENCE_LABEL:
                background_volume = np.random.uniform(0, 1)
            elif np.random.uniform(0, 1) < training_parameters["background_frequency"]:
                background_volume = np.random.uniform(
                    0, training_parameters["background_volume"]
                )
            else:
                background_volume = 0
        else:
            background_reshaped = np.zeros(
                [self.data_processing_parameters["desired_samples"], 1]
            )
            background_volume = 0

        data_augmentation_parameters["background_noise"] = background_reshaped
        data_augmentation_parameters["background_volume"] = background_volume

        # For silence samples, remove any sound
        if sample["label"] == SILENCE_LABEL:
            data_augmentation_parameters["foreground_volume"] = 0
        else:
            data_augmentation_parameters["foreground_volume"] = 1

        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        # Load data
        try:
            sf_loader, _ = sf.read(data_augmentation_parameters["wav_filename"])
            wav_file = torch.Tensor(np.array([sf_loader]))
        except Exception as e:
            print(f"Error loading WAV file '{data_augmentation_parameters['wav_filename']}': {e}")
            wav_file = None  # Explicitly set wav_file to None if reading fails

        # Ensure wav_file is assigned before further processing
        if wav_file is None:
            raise ValueError(f"wav_file is None. Failed to load '{data_augmentation_parameters['wav_filename']}'.")

        # Ensure data length is equal to the number of desired samples
        try:
            desired_samples = self.data_processing_parameters["desired_samples"]
            if len(wav_file[0]) < desired_samples:
                wav_file = torch.nn.ConstantPad1d(
                    (
                        0,
                        desired_samples - len(wav_file[0]),
                    ),
                    0,
                )(wav_file[0])
            else:
                wav_file = wav_file[0][:desired_samples]
        except Exception as e:
            print(f"Error processing WAV file data length: {e}")
            raise

        # Scale foreground
        try:
            scaled_foreground = torch.mul(
                wav_file, data_augmentation_parameters["foreground_volume"]
            )
        except Exception as e:
            print(f"Error scaling foreground: {e}")
            raise


        # Padding wrt the time shift offset
        pad_tuple = tuple(data_augmentation_parameters["time_shift_padding"][0])
        padded_foreground = torch.nn.ConstantPad1d(pad_tuple, 0)(scaled_foreground)
        sliced_foreground = padded_foreground[
            data_augmentation_parameters["time_shift_offset"][
                0
            ] : data_augmentation_parameters["time_shift_offset"][0]
            + self.data_processing_parameters["desired_samples"]
        ]

        # Mix in background noise
        background_mul = torch.mul(
            torch.Tensor(data_augmentation_parameters["background_noise"][:, 0]),
            data_augmentation_parameters["background_volume"],
        )
        background_add = torch.add(background_mul, sliced_foreground)

        # Compute MFCCs - PyTorch
        # melkwargs={ 'n_fft':1024, 'win_length':self.data_processing_parameters['window_size_samples'], 'hop_length':self.data_processing_parameters['window_stride_samples'],
        #        'f_min':20, 'f_max':4000, 'n_mels':40}
        # mfcc_transformation = torchaudio.transforms.MFCC(n_mfcc=self.data_processing_parameters['feature_bin_count'], sample_rate=self.data_processing_parameters['desired_samples'], melkwargs=melkwargs, log_mels=True, norm='ortho')
        # data = mfcc_transformation(background_add)
        # data_result[i] = data[:,:self.data_processing_parameters['spectrogram_length']].numpy().transpose()

        # Compute MFCCs - TensorFlow (matching C-based implementation)
        tf_data = tf.convert_to_tensor(background_add.numpy(), dtype=tf.float32)
        tf_stfts = tf.signal.stft(
            tf_data,
            frame_length=self.data_processing_parameters["window_size_samples"],
            frame_step=self.data_processing_parameters["window_stride_samples"],
            fft_length=1024,
        )
        tf_spectrograms = tf.abs(tf_stfts)
        power = True
        if power:
            tf_spectrograms = tf_spectrograms**2
        num_spectrogram_bins = tf_stfts.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            40,
            num_spectrogram_bins,
            self.data_processing_parameters["desired_samples"],
            20,
            4000,
        )
        tf_spectrograms = tf.cast(tf_spectrograms, tf.float32)
        tf_mel_spectrograms = tf.tensordot(
            tf_spectrograms, linear_to_mel_weight_matrix, 1
        )
        tf_mel_spectrograms.set_shape(
            tf_spectrograms.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]
            )
        )
        tf_log_mel = tf.math.log(tf_mel_spectrograms + 1e-6)
        tf_mfccs = tf.signal.mfccs_from_log_mel_spectrograms(tf_log_mel)[
            ..., : self.data_processing_parameters["feature_bin_count"]
        ]
        mfcc = torch.Tensor(tf_mfccs.numpy())
        data_result = mfcc

        # Shift data in [0, 255] interval to match Dory request for uint8 inputs
        data_result = np.clip(data_result + 128, 0, 255)
        label_result = sample["label"]
        return label_result, data_result

    def compute_mfccs_all(self, dataset_path_rawdio, dataset_path_raw_mfccs):
        word_paths = [f.path for f in os.scandir(dataset_path_rawdio) if f.is_dir()]

        for word_path in word_paths:
            # path = os.path.join(dataset_path_raw_mfccs, raw_mfccs_path)
            if word_path[-1] != '_':
                somepath = dataset_path_raw_mfccs + "/" + word_path.split("/")[-1]
                os.mkdir(somepath)
                wav_paths = [f.path for f in os.scandir(word_path)]

                for wav_path in wav_paths:
                    someotherpath = (
                        somepath + "/" + wav_path.split("/")[-1].split(".wav")[0]
                    )
                    _, mfccs = self.compute_mfccs_sample(wav_path)
                    mfccs.numpy().tofile(someotherpath)
            else:
                print("skipping word: " + word_path)


if __name__ == "__main__":

    speech_commands_path = "./dataset"
    raw_mfccs_path = "./dataset_mfccs_raw"

    path = os.path.join(".", raw_mfccs_path)

    # Remove the specified
    # file path
    if os.path.isdir(raw_mfccs_path):

        shutil.rmtree(raw_mfccs_path)
        os.mkdir(path)
    else:
        os.mkdir(path)

    ap_obj = AudioProcessor(training_parameters, data_processing_parameters)
    ap_obj.compute_mfccs_all(speech_commands_path, raw_mfccs_path)
