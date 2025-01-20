STEP_DO_TRAIN           = True
STEP_DO_PROCESS_MFCCS   = True  # New step
STEP_DO_EXPORT_MODEL    = False
CHECKPOINT_PATH         = './model_acc_94.53125.pth'
CLASSES                 = 12
EXPORT_OUTPUT_DIR_PATH  = '../simulation/exported_models/'
EXPORT_OUTPUT_NAME      = 'export_params_nclass_' + str(CLASSES) + '.csv'
EXPORT_OUTPUT_PATH      = EXPORT_OUTPUT_DIR_PATH + EXPORT_OUTPUT_NAME
MFCCS_INPUT_PATH        = '../dataset_mfccs_raw/yes/d21fd169_nohash_0'  # Path to MFCCs binary file
MFCCS_OUTPUT_PATH       = './output_mfccs.bin' # Path to save model output

import torch
import dataset
import torch.nn as nn
import os
import time
import csv
from torchsummary import summary
from model import DSCNN
from utils import remove_txt, parameter_generation
from train import Train
import numpy as np

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)

# Model generation
model = DSCNN(use_bias=True)
model.to(device)

if STEP_DO_TRAIN:
    training_parameters, data_processing_parameters = parameter_generation()
    audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/Valid/Test):", train_size, "/", valid_size, "/", test_size)

    print("Printing model summary...")
    summary(model, (1, 49, data_processing_parameters['feature_bin_count']))
    dummy_input = torch.rand(1, 1, 49, data_processing_parameters['feature_bin_count']).to(device)

    print("Initializing training environment...")
    trainining_environment = Train(audio_processor, training_parameters, model, device)

    print("Removing stored inputs and activations...")
    remove_txt()

    print("Starting training...")
    start = time.time()
    trainining_environment.train(model)
    print('Finished Training on GPU in {:.2f} seconds'.format(time.time() - start))

if STEP_DO_EXPORT_MODEL:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    if not os.path.isdir(EXPORT_OUTPUT_DIR_PATH):
        os.mkdir(EXPORT_OUTPUT_DIR_PATH)

    try:
        with open(EXPORT_OUTPUT_PATH, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Name", "Shape", "Values"])
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.writerow([name, list(param.shape), param.detach().cpu().numpy().flatten().tolist()])
        print(f"Model parameters exported successfully to {EXPORT_OUTPUT_PATH}.")
    except Exception as e:
        print(f"Error exporting model parameters: {e}")


# Recursive function to register hooks on all layers
def register_hooks(model):
    layer_outputs = []
    layer_names = []

    def hook_fn(module, input, output):
        layer_outputs.append(output.cpu().numpy())
        layer_names.append(str(module))  # Capture the layer name

    hooks = []

    # Traverse all modules (including nested layers)
    def register(module):
        if isinstance(module, nn.Module):
            for name, child in module.named_children():
                # Register hooks for all layers of interest (Conv2d, Linear, BatchNorm, ReLU, etc.)
                if isinstance(child, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.Dropout, nn.AdaptiveAvgPool2d)):
                    hook = child.register_forward_hook(hook_fn)
                    hooks.append(hook)
                # Recursively register hooks for child modules
                register(child)

    # Start registering hooks from the top-level model
    register(model)

    return layer_outputs, layer_names, hooks

if STEP_DO_PROCESS_MFCCS:
    try:
        print("Processing MFCCs input...")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        
        mfccs_data = np.fromfile(MFCCS_INPUT_PATH, dtype=np.float32)
        mfccs_tensor = torch.tensor(mfccs_data).reshape(1, 1, 49, -1).to(device)

        model.eval()
        layer_outputs, layer_names, hooks = register_hooks(model)  # Register hooks

        with torch.no_grad():
            output = model(mfccs_tensor)

        output_numpy = output.cpu().numpy()

        # Save the model output to a CSV file
        csv_output_path = os.path.splitext(MFCCS_OUTPUT_PATH)[0] + '.csv'
        with open(csv_output_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Output Values"])  # Header for final output
            for value in output_numpy.flatten():
                writer.writerow([value])

            # Write layer outputs along with their names
            for idx, (layer_output, layer_name) in enumerate(zip(layer_outputs, layer_names)):
                writer.writerow([f"{layer_name} Output"])  # Header for each layer's output
                for layer_value in layer_output.flatten():
                    writer.writerow([layer_value])

        # Remove hooks
        for hook in hooks:
            hook.remove()

        print(f"Processed MFCCs and layer outputs saved to {csv_output_path}.")
    
    except Exception as e:
        print(f"Error processing MFCCs file: {e}")