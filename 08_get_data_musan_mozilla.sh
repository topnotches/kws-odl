#!/bin/bash

# Base directory for musan
BASE_DIR="dataset_xvector_mozilla/musan"

# Create the musan directory structure
mkdir -p "$BASE_DIR"

# Download and extract musan dataset
MUSAN_URL="https://openslr.elda.org/resources/17/musan.tar.gz"
MUSAN_ARCHIVE="musan.tar.gz"

# Download the musan dataset
if [ ! -f "$MUSAN_ARCHIVE" ]; then
    echo "Downloading musan dataset..."
    curl -o "$MUSAN_ARCHIVE" "$MUSAN_URL"
else
    echo "Musan dataset archive already exists. Skipping download."
fi

# Extract the archive
if [ -f "$MUSAN_ARCHIVE" ]; then
    echo "Extracting musan dataset..."
    tar -xzf "$MUSAN_ARCHIVE" -C "$BASE_DIR" --strip-components=1
    echo "Musan dataset extracted to $BASE_DIR."
else
    echo "Error: Musan archive not found!"
    exit 1
fi

# Print completion message
echo "Musan directory is ready in $BASE_DIR."
