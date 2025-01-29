#!/usr/bin/env python3
import os
import shutil
import random
from pathlib import Path
from collections import Counter

def create_mozilla_layout(validated_file, clips_dir, output_path):
    # Ensure the output directories exist
    dev_path = output_path / "train"
    test_path = output_path / "test"
    all_path = output_path / "all"

    # Remove contents of dev and test directories if they exist
    if dev_path.exists():
        shutil.rmtree(dev_path)
    if test_path.exists():
        shutil.rmtree(test_path)
    if all_path.exists():
        shutil.rmtree(all_path)

    dev_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    all_path.mkdir(parents=True, exist_ok=True)

    # Read the validated TSV file and prepare speaker ids
    client_ids = []
    files = []
    current_client_id = ""

    # Count how many samples each client_id has
    client_sample_count = Counter()

    with open(validated_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            bobbert = line.strip().split("\t")
            client_id = bobbert[0]
            path = bobbert[1]
            files.append((client_id, path))
            client_sample_count[client_id] += 1

    # Sort client_ids based on the sample count (descending order)
    sorted_client_ids = [client_id for client_id, _ in client_sample_count.most_common()]

    # Limit to the top 10,000 client_ids
    top_client_ids = sorted_client_ids[:10000]

    # Shuffle client ids for randomness
    random.shuffle(top_client_ids)

    # Split client ids into 90% for train and 10% for test
    split_index = int(len(top_client_ids) * 0.9)
    dev_speakers = set(top_client_ids[:split_index])
    test_speakers = set(top_client_ids[split_index:])

    # Create a map for formatted speaker IDs
    formatted_client_ids = {sid: f"uid{int(i + 1):06d}" for i, sid in enumerate(top_client_ids)}

    # Process each file and create the appropriate layout
    for client_id, path in files:
        if client_id not in top_client_ids:
            continue  # Skip files that don't belong to the top 10,000 clients
        
        # Create directories for each client_id
        formatted_client_id = formatted_client_ids[client_id]

        # Define paths for dev and test
        if client_id in dev_speakers:
            speaker_dir = dev_path / formatted_client_id
        else:
            speaker_dir = test_path / formatted_client_id

        # Create the speaker directory if it doesn't exist
        speaker_dir.mkdir(parents=True, exist_ok=True)

        # Define the source and destination paths
        input_file = Path(clips_dir) / path
        output_file = speaker_dir / f"{formatted_client_id}_{path.split('/')[-1]}"

        # Convert MP3 to WAV and copy the file
        if input_file.exists():
            shutil.copy(input_file, output_file)
        else:
            print(f"File not found: {input_file}")

    # Generate the verification file
    generate_veri_test2(dev_path, test_path, output_path)

def generate_veri_test2(dev_path, test_path, output_path):
    # Path for the verification file
    veri_test_path = output_path / "veri_test2.txt"

    with veri_test_path.open("w") as veri_file:
        test_files = list(test_path.glob("**/*.mp3"))
        random.shuffle(test_files)

        # Create pairs for verification
        for file1 in test_files:
            speaker_id1 = file1.parts[-2]

            # Same speaker pairs
            same_files = [
                f for f in test_files if f.parts[-2] == speaker_id1 and f != file1
            ]
            random.shuffle(same_files)
            same_files = same_files[:2]  # Limit to 20 pairs per speaker

            for same_file in same_files:
                veri_file.write(
                    f"1 {file1.parent.parent.name}/{file1.parent.name}/{file1.name} {same_file.parent.parent.name}/{same_file.parent.name}/{same_file.name}\n"
                )

            # Different speaker pairs
            diff_files = [f for f in test_files if f.parts[-2] != speaker_id1]
            random.shuffle(diff_files)
            diff_files = diff_files[:2]  # Limit to 20 pairs per speaker

            for diff_file in diff_files:
                veri_file.write(
                    f"0 {file1.parent.parent.name}/{file1.parent.name}/{file1.name} {diff_file.parent.parent.name}/{diff_file.parent.name}/{diff_file.name}\n"
                )

if __name__ == "__main__":
    # Path to the validated.tsv file and clips directory
    validated_file = "../dataset_mozilla/cv-corpus-20.0-2024-12-06/en/validated.tsv"
    clips_dir = "../dataset_mozilla/cv-corpus-20.0-2024-12-06/en/clips"

    # Path to the output directory
    output_path = Path("./dataset_xvector_mozilla/VoxCeleb")

    create_mozilla_layout(validated_file, clips_dir, output_path)

    print(f"Converted dataset to new layout with train/test split in {output_path}")
