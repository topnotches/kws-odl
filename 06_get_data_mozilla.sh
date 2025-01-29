#!/bin/bash

# Define variables
DATASET_VERSION="cv-corpus-20.0-2024-12-11"  # Update this if the version changes
LANGUAGE="en"  # Change this to your desired language code (e.g., "fr" for French)
BASE_DIR=".."
DATASET_DIR="$BASE_DIR/dataset_mozilla"
DATASET_URL="https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/${DATASET_VERSION}/${LANGUAGE}.tar.gz"
DATASET_ARCHIVE="cv-corpus-20.0-2024-12-06-en.tar.gz"

# Create dataset directory if it doesn't exist
echo "Creating dataset directory at $DATASET_DIR..."
mkdir -p "$DATASET_DIR"

# Change to the dataset directory

# Download the dataset if not already present
if [ ! -f "$DATASET_ARCHIVE" ]; then
    echo "Downloading Mozilla Common Voice Corpus 20.0 for language $LANGUAGE..."
    wget "$DATASET_URL"
else
    echo "Dataset archive already exists: $DATASET_ARCHIVE"
fi

# Extract the dataset
echo "Extracting dataset..."
tar -xvzf "$DATASET_ARCHIVE" -C "$DATASET_DIR"

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
echo "Make sure your code's search path matches: $DATASET_DIR/${DATASET_VERSION}/${LANGUAGE}/*.tsv and $DATASET_DIR/${DATASET_VERSION}/${LANGUAGE}/clips/*.mp3"

# Done
echo "Setup completed!"