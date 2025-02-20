
import torch
import dataset
import torch.nn as nn
import os
import torch.ao.quantization as quant
import time
import csv
from torchsummary import summary
from model import DSCNN, DSCNN_fusable
from utils import *

import shutil
from train import Train
import numpy as np

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)
model = DSCNN(use_bias=True)
model.to(device)

if (STEP_DO_QAT_TRAIN):
    
    model_unprep = DSCNN_fusable(use_bias=True)

    model_unprep.eval()

    
    model_unprep_fused = torch.ao.quantization.fuse_modules_qat(
        model_unprep,
        quant_fuse_list
    )


    for name, module in model_unprep_fused.named_modules():
        for key, qconfig in qat_configs.items():
            if name.startswith(key):  
                module.qconfig = qconfig

    

    
    model = torch.ao.quantization.prepare_qat(model_unprep_fused.train())

    print("Model qconfig before training:", model.qconfig)
    model.to(device)
else:
    model = DSCNN(use_bias=True)
    model.to(device)
if STEP_DO_TRAIN:
    
    if not os.path.isdir(CHECKPOINT_SAVE_PATH):
        os.mkdir(CHECKPOINT_SAVE_PATH)
        shutil.copyfile('./utils.py', CHECKPOINT_SAVE_PATH+'/000_utils_setup.log', follow_symlinks = True)
    else:
        assert False, "HELLOOOOO. THE DIRECTORY IS ALREADY THERE, FOKER!"
    training_parameters, data_processing_parameters = parameter_generation()
    audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/Valid/Test):", train_size, "/", valid_size, "/", test_size)

    print("Printing model summary...")
    dummy_input = torch.rand(1, 1, 49, data_processing_parameters['feature_bin_count']).to(device)

    print("Initializing training environment...")
    trainining_environment = Train(audio_processor, training_parameters, model, device)

    print("Removing stored inputs and activations...")
    remove_txt()

    print("Starting training...")
    start = time.time()

    trainining_environment.train(model)
    print('Finished Training on GPU in {:.2f} seconds'.format(time.time() - start))

    if STEP_DO_QAT_TRAIN:
        model.eval() 
        for name, module in model.named_modules():
            if hasattr(module, 'weight_fake_quant'):
                print(f"{name} weight observer dtype: {module.weight_fake_quant.dtype}")
        model_int8 = torch.ao.quantization.convert(model.cpu())
    
        torch.save(model_int8.state_dict(), "quantized_model_complete.pth")

if STEP_DO_EXPORT_MODEL:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    if not os.path.isdir(EXPORT_OUTPUT_DIR_PATH_FLOAT):
        os.mkdir(EXPORT_OUTPUT_DIR_PATH_FLOAT)

    try:
        with open(EXPORT_OUTPUT_PATH_FLOAT, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Name", "Shape", "Values"])

            processed_bn_layers = set()  # Track processed BatchNorm layers

            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.writerow([name, list(param.shape), param.detach().cpu().numpy().flatten().tolist()])

                    # Extract layer name (remove trailing ".weight" or ".bias")
                    bn_layer_name = name.rsplit('.', 1)[0]
                    if bn_layer_name not in processed_bn_layers and bn_layer_name in dict(model.named_modules()) and  name.rsplit('.', 1)[1] == "bias":
                        module = dict(model.named_modules())[bn_layer_name]
                        if isinstance(module, nn.BatchNorm2d):
                            writer.writerow([f"{bn_layer_name}.running_mean", list(module.running_mean.shape), 
                                            module.running_mean.cpu().numpy().flatten().tolist()])
                            writer.writerow([f"{bn_layer_name}.running_var", list(module.running_var.shape), 
                                            module.running_var.cpu().numpy().flatten().tolist()])
                            processed_bn_layers.add(bn_layer_name)  # Mark as processed to avoid duplicates

        print(f"Model parameters and BatchNorm stats exported successfully to {EXPORT_OUTPUT_PATH_FLOAT}.")

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
                if isinstance(child, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.Dropout, nn.AvgPool2d)):
                    hook = child.register_forward_hook(hook_fn)
                    hooks.append(hook)
                # Recursively register hooks for child modules
                register(child)

    # Start registering hooks from the top-level model
    register(model)

    return layer_outputs, layer_names, hooks

if STEP_DO_PROCESS_MFCCS:

    # Main script to process MFCCs from a list of file paths
    print("Processing MFCCs input...")
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

        # Process all MFCCs in the batch
        batch_data = []
        for path in MFCCS_INPUT_PATHS:  # list of MFCC file paths
            mfcc_data = np.fromfile(path, dtype=np.float32)
            
            # Check the shape of the input MFCC data and reshape accordingly
            # Assuming MFCC data is a 1D array that needs to be reshaped into 4D tensor
            # Example reshape to (batch_size, channels, height, width) -> [1, 1, 49, width]
            mfcc_tensor = torch.tensor(mfcc_data).reshape( 1, 49, -1).to(device)
            
            batch_data.append(mfcc_tensor)

        # Stack all the tensors in the batch
        batch_tensor = torch.stack(batch_data)

        model.eval()
        layer_outputs, layer_names, hooks = register_hooks(model)  # Register hooks

        with torch.no_grad():
            output = model(batch_tensor)  # Process the entire batch

        output_numpy = output.cpu().numpy()

        # Save the model output to a CSV file
        csv_output_path = os.path.splitext(MFCCS_OUTPUT_PATH)[0] + '.csv'
        with open(csv_output_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Output Values"])  # Header for final output

            # Write all outputs in the batch to the CSV
            for output_batch in output_numpy:
                for value in output_batch.flatten():
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
        print(f"Error processing MFCCs files: {e}")
