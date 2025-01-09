#!/bin/bash

# Define variables
DATASET_VERSION="v0.02"
BASE_DIR="."
DATASET_DIR="$BASE_DIR/dataset"
DATASET_URL="http://download.tensorflow.org/data/speech_commands_${DATASET_VERSION}.tar.gz"
DATASET_ARCHIVE="speech_commands_${DATASET_VERSION}.tar.gz"

# Create dataset directory if it doesn't exist
echo "Creating dataset directory at $DATASET_DIR..."
mkdir -p "$DATASET_DIR"

# Change to the dataset directory
cd "$DATASET_DIR" || exit

# Download the dataset if not already present
if [ ! -f "$DATASET_ARCHIVE" ]; then
    echo "Downloading Speech Commands dataset version $DATASET_VERSION..."
    wget "$DATASET_URL"
else
    echo "Dataset archive already exists: $DATASET_ARCHIVE"
fi

# Extract the dataset
echo "Extracting dataset..."
tar -xvzf "$DATASET_ARCHIVE"

# Verify extraction
if [ $? -eq 0 ]; then
    echo "Dataset extracted successfully to $DATASET_DIR."
else
    echo "Failed to extract dataset. Please check the archive file."
    exit 1
fi

# Display directory structure
echo "Directory structure in $DATASET_DIR:"
ls -lh "$DATASET_DIR"

# Provide feedback for script usage in code
echo "Make sure your code's search path matches: $DATASET_DIR/speech_commands_${DATASET_VERSION}/*/*.wav"

# Done
echo "Setup completed!"
