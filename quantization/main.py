STEP_DO_QAT_TRAIN       = True
STEP_DO_TRAIN           = True
STEP_DO_EXPORT_MODEL    = False
STEP_DO_PROCESS_MFCCS   = False
CHECKPOINT_PATH         = 'none'
CLASSES                 = 10
EXPORT_OUTPUT_DIR_PATH  = '../simulation/exported_models/'
EXPORT_OUTPUT_NAME      = 'export_params_nclass_' + str(CLASSES) + '.csv'
EXPORT_OUTPUT_NAME_QAT      = 'qat_export_params_nclass_' + str(CLASSES) + '.csv'
EXPORT_OUTPUT_PATH      = EXPORT_OUTPUT_DIR_PATH + EXPORT_OUTPUT_NAME
MFCCS_INPUT_PATHS        = ['../dataset_mfccs_raw/yes/d21fd169_nohash_0',
                            '../dataset_mfccs_raw/yes/d21fd169_nohash_1']  # Path(s) to MFCCs binary file
MFCCS_OUTPUT_PATH       = './output_mfccs.bin' # Path to save model output

import torch
import dataset
import torch.nn as nn
import os
import torch.ao.quantization as quant
import time
import csv
from torchsummary import summary
from model import DSCNN, DSCNN_fusable
from utils import remove_txt, parameter_generation
from train import Train
import numpy as np

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)
model = DSCNN(use_bias=True)
model.to(device)
def get_qconfig8(bits):
    qmax = 2**bits - 1
    qmin_signed = -2**(bits-1)
    qmax_signed = 2**(bits-1) - 1

    return quant.QConfig(
        activation=quant.FakeQuantize.with_args(
            observer=quant.MovingAverageMinMaxObserver,
            quant_min=0, quant_max=qmax,  # Unsigned activation
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine
        ),
        weight=quant.FakeQuantize.with_args(
            observer=quant.MovingAverageMinMaxObserver,
            quant_min=qmin_signed, quant_max=qmax_signed,  # Signed weight
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric
        )
    )
def get_qconfig16(bits):
    qmax = 2**bits - 1
    qmin_signed = -2**(bits-1)
    qmax_signed = 2**(bits-1) - 1

    return quant.QConfig(
        activation=quant.FakeQuantize.with_args(
            observer=quant.MovingAverageMinMaxObserver,
            quant_min=0, quant_max=qmax,  # Unsigned activation
            dtype=torch.qint32,
            qscheme=torch.per_tensor_affine
        ),
        weight=quant.FakeQuantize.with_args(
            observer=quant.MovingAverageMinMaxObserver,
            quant_min=qmin_signed, quant_max=qmax_signed,  # Signed weight
            dtype=torch.qint32,
            qscheme=torch.per_tensor_symmetric
        )
    )
if (STEP_DO_QAT_TRAIN):
    
    model_unprep = DSCNN_fusable(use_bias=True)

    model_unprep.eval()

    
    model_unprep_fused = torch.ao.quantization.fuse_modules_qat(
        model_unprep,
        [
            ["ConvBNReLU1.0","ConvBNReLU1.1","ConvBNReLU1.2"],
            ["ConvBNReLU2.0","ConvBNReLU2.1","ConvBNReLU2.2"],
            ["ConvBNReLU3.0","ConvBNReLU3.1","ConvBNReLU3.2"],
            ["ConvBNReLU4.0","ConvBNReLU4.1","ConvBNReLU4.2"],
            ["ConvBNReLU5.0","ConvBNReLU5.1","ConvBNReLU5.2"],
            ["ConvBNReLU6.0","ConvBNReLU6.1","ConvBNReLU6.2"],
            ["ConvBNReLU7.0","ConvBNReLU7.1","ConvBNReLU7.2"],
            ["ConvBNReLU8.0","ConvBNReLU8.1","ConvBNReLU8.2"],
            ["ConvBNReLU9.0","ConvBNReLU9.1","ConvBNReLU9.2"],
        ]
    )


    
    qat_configs = {
        "ConvBNReLU1.0": get_qconfig8(8),
        "ConvBNReLU2.0": get_qconfig8(4),
        "ConvBNReLU3.0": get_qconfig8(4),
        "ConvBNReLU4.0": get_qconfig8(4),
        "ConvBNReLU5.0": get_qconfig8(4),
        "ConvBNReLU6.0": get_qconfig8(4),
        "ConvBNReLU7.0": get_qconfig8(4),
        "ConvBNReLU8.0": get_qconfig8(4),
        "ConvBNReLU9.0": get_qconfig8(4),
        "fc1": get_qconfig8(8),
    }
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

    # Train the model now that it's prepared for QAT
    trainining_environment.train(model)
    print('Finished Training on GPU in {:.2f} seconds'.format(time.time() - start))

    # Convert the model to quantized format after training
    if STEP_DO_QAT_TRAIN:
        model.eval()  # Ensure it's in eval mode before converting
        for name, module in model.named_modules():
            if hasattr(module, 'weight_fake_quant'):
                print(f"{name} weight observer dtype: {module.weight_fake_quant.dtype}")
        model_int8 = torch.ao.quantization.convert(model.cpu())
    
        # Print layer types to check if they are quantized
        print("\nModel structure after quantization:")
        print(model_int8)  

        # Print parameters and check if they are quantized
        print("\nModel parameters after quantization:")
        for name, param in model_int8.named_parameters():
            print(f"{name}: dtype={param.dtype}, shape={param.shape}")
        for name, module in model_int8.named_modules():
            if hasattr(module, 'weight'):
                print(f"{name} - Quantized Weight:")
                print(module.weight().int_repr())  # Get int values
        torch.save(model_int8.state_dict(), "quantized_model_complete.pth")

if STEP_DO_EXPORT_MODEL:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    if not os.path.isdir(EXPORT_OUTPUT_DIR_PATH):
        os.mkdir(EXPORT_OUTPUT_DIR_PATH)

    try:
        with open(EXPORT_OUTPUT_PATH, mode='w', newline='') as csv_file:
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

        print(f"Model parameters and BatchNorm stats exported successfully to {EXPORT_OUTPUT_PATH}.")

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
