import torch
import numpy as np

# Path to the quantized model and input GSC file
QUANTIZED_MODEL_PATH = "quantized_model_complete.pth"
GSC_FILE_PATH = "../dataset_mfccs_raw/yes/d21fd169_nohash_0"

# Load the quantized model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(QUANTIZED_MODEL_PATH, map_location=device)
model.eval()  # Set model to evaluation mode

# Print model parameters
print("\nModel Parameters:")
for name, param in model.named_parameters():
    print(f"{name}: shape {param.shape}")
    print(param.detach().cpu().numpy().flatten()[:])  # Print first 10 values for brevity

# Dictionary to store activations
activations = {}

# Function to hook into layers and store activations
def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output.cpu().numpy()
    return hook

# Register hooks for all convolutional and fully connected layers
hooks = []
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU, torch.nn.BatchNorm2d)):
        hooks.append(module.register_forward_hook(hook_fn(name)))

# Load the GSC MFCC data
mfcc_data = np.fromfile(GSC_FILE_PATH, dtype=np.float32)

# Reshape the MFCC data to match the expected input shape
# Assuming the expected input shape is (batch_size, channels, height, width)
mfcc_tensor = torch.tensor(mfcc_data).reshape(1, 1, 49, -1).to(device)

# Run inference
with torch.no_grad():
    output = model(mfcc_tensor)

# Convert output to numpy and print final output
output_numpy = output.cpu().numpy()
print("\nModel Output:", output_numpy)

# Print intermediate activations
print("\nIntermediate Activations:")
for layer_name, activation in activations.items():
    print(f"{layer_name}: shape {activation.shape}")
    print(activation.flatten()[:])  # Print first 10 values for brevity

# Remove hooks
for hook in hooks:
    hook.remove()
