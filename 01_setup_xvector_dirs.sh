#!/bin/bash

# Define the base directory
BASE_DIR="dataset_xvector"

# Create the base directory
mkdir -p "$BASE_DIR"

# Create VoxCeleb structure
mkdir -p "$BASE_DIR/VoxCeleb/vox1_dev_wav"
mkdir -p "$BASE_DIR/VoxCeleb/vox1_test_wav"

# Create musan structure
mkdir -p "$BASE_DIR/musan/music"
mkdir -p "$BASE_DIR/musan/noise"
mkdir -p "$BASE_DIR/musan/speech"

# Create rir_noises structure
mkdir -p "$BASE_DIR/rir_noises/simulated_rirs"

# Output a message indicating completion
echo "Directory structure created under $BASE_DIR"
