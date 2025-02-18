import torch
import numpy as np

import torch.ao.quantization as quant
# Path to the quantized model and input GSC file
QUANTIZED_MODEL_PATH = "quantized_model_complete.pth"
GSC_FILE_PATH = "../dataset_mfccs_raw/yes/d21fd169_nohash_0"
import torch
from model import DSCNN_fusable  # Ensure you import the model class

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

# Create an instance of your model
model_unprep = DSCNN_fusable(use_bias=True)

model_unprep.eval()

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

model.eval()  # Ensure it's in eval mode before converting
for name, module in model.named_modules():
    if hasattr(module, 'weight_fake_quant'):
        print(f"{name} weight observer dtype: {module.weight_fake_quant.dtype}")
model_int8 = torch.ao.quantization.convert(model.cpu())
# Load weights

model_int8.load_state_dict(torch.load(QUANTIZED_MODEL_PATH, map_location=device))
model_int8.eval()


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
#for name, module in model_int8.named_modules():
#    if hasattr(module, 'weight'):
#        weight = module.weight()
#        print(f"\n{name} - Quantized Weight (int representation):")
#        print(weight.int_repr())  # Get the raw int8/int4 values
#        
#        print(f"{name} - Scale: {weight.q_scale()}")
#        print(f"{name} - Zero Point: {weight.q_zero_point()}")
#
#        print(f"{name} - Dequantized Weight:")
#        print(weight.dequantize())  # Convert back to float

# Remove hooks
for hook in hooks:
    hook.remove()
