#!/bin/bash

# Base directory for RIRs and noises
BASE_DIR="dataset_xvector/rir_noises"

# Create the rir_noises directory structure
mkdir -p "$BASE_DIR"

# URL and archive for RIRs and noises
RIR_URL="https://openslr.elda.org/resources/28/rirs_noises.zip"
RIR_ARCHIVE="rirs_noises.zip"

# Download the RIRs and noises dataset
if [ ! -f "$RIR_ARCHIVE" ]; then
    echo "Downloading RIRs and noises dataset..."
    curl -o "$RIR_ARCHIVE" "$RIR_URL"
else
    echo "RIRs and noises archive already exists. Skipping download."
fi

# Extract the archive
if [ -f "$RIR_ARCHIVE" ]; then
    echo "Extracting RIRs and noises dataset..."
    unzip -q "$RIR_ARCHIVE" -d "$BASE_DIR"
    echo "RIRs and noises dataset extracted to $BASE_DIR."
else
    echo "Error: RIRs and noises archive not found!"
    exit 1
fi

# Print completion message
echo "RIRs and noises directory is ready in $BASE_DIR."
