#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
import random

def create_voxceleb_layout(speech_commands_path, output_path):
    # Define paths
    speech_commands_path = Path(speech_commands_path)
    output_path = Path(output_path)

    # Ensure the output directories exist
    dev_path = output_path / "vox1_dev_wav"
    test_path = output_path / "vox1_test_wav"

    # Remove contents of dev and test directories if they exist
    if dev_path.exists():
        shutil.rmtree(dev_path)
    if test_path.exists():
        shutil.rmtree(test_path)
    if output_path.exists():
        for item in output_path.iterdir():
            if item.is_dir() and item.name != "vox1_dev_wav" and item.name != "vox1_test_wav":
                shutil.rmtree(item)

    dev_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # Get all subdirectories (keywords in Speech Commands)
    keywords = [d for d in speech_commands_path.iterdir() if d.is_dir()]

    # Extract unique speaker IDs
    speaker_ids = set()
    for keyword in keywords:
        if keyword.name != "dataset/_background_noise_":
            audio_files = list(keyword.glob("*.wav"))
            for audio_file in audio_files:
                speaker_id = audio_file.stem.split("_")[0]
                speaker_ids.add(speaker_id)

    speaker_ids = list(speaker_ids)
    random.shuffle(speaker_ids)  # Shuffle the speaker IDs for randomness

    # Split speaker IDs into 90% for dev and 10% for test
    split_index = int(len(speaker_ids) * 0.9)
    dev_speakers = set(speaker_ids[:split_index])
    test_speakers = set(speaker_ids[split_index:])

    for keyword in keywords:
        audio_files = list(keyword.glob("*.wav"))

        for audio_file in audio_files:
            # Extract the speaker ID from the filename
            speaker_id = audio_file.stem.split("_")[0]

            # Append the keyword (word) to the file name
            word = keyword.name
            new_file_name = f"{audio_file.stem}_{word}{audio_file.suffix}"

            # Determine whether the file belongs to dev or test set
            if speaker_id in dev_speakers:
                speaker_dir = dev_path / f"id{speaker_id}"
            else:
                speaker_dir = test_path / f"id{speaker_id}"

            # Create the directory if it doesn't exist
            speaker_dir.mkdir(parents=True, exist_ok=True)

            # Copy the audio file to the appropriate directory
            shutil.copy(audio_file, speaker_dir / new_file_name)
    # Generate the veri_test2.txt file

    # Generate the veri_test2.txt file
    veri_test_path = output_path / "veri_test2.txt"
    with veri_test_path.open("w") as veri_file:
        test_files = list(test_path.glob("**/*.wav"))
        used_speakers = set()

        # Iterate over test files and create multiple pairs per speaker
        for file1 in test_files:
            speaker_id = file1.parts[-2]

            if speaker_id not in used_speakers:
                # Create multiple pairs for the speaker
                same_files = [f for f in test_files if f.parts[-2] == speaker_id and f != file1]
                diff_files = [f for f in test_files if f.parts[-2] != speaker_id and f.stem.split("_")[1] == file1.stem.split("_")[1]]

                # Limit to 5 pairs each to avoid excessive runtime
                same_files = same_files[:20]
                diff_files = diff_files[:20]

                for same_file in same_files:
                    veri_file.write(f"1 {file1.parent.name}/{file1.name} {same_file.parent.name}/{same_file.name}\n")

                for diff_file in diff_files:
                    veri_file.write(f"0 {file1.parent.name}/{file1.name} {diff_file.parent.name}/{diff_file.name}\n")

                used_speakers.add(speaker_id)
if __name__ == "__main__":
    # Path to the Speech Commands dataset (v0.0.2)
    speech_commands_path = "./dataset"

    # Path to the output directory
    output_path = Path("./dataset_xvector/VoxCeleb")

    create_voxceleb_layout(speech_commands_path, output_path)

    print(f"Converted Speech Commands dataset to VoxCeleb layout with train/dev split by speaker ID in {output_path}")
