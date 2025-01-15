STEP_DO_TRAIN           = False
STEP_DO_EXPORT_MODEL    = True
CHECKPOINT_PATH         = './model_acc_94.53125.pth'
CLASSES                 = 12
EXPORT_OUTPUT_DIR_PATH  = '../simulation/exported_models/'
EXPORT_OUTPUT_NAME      = 'export_params_nclass_' + str(CLASSES) + '.csv'
EXPORT_OUTPUT_PATH      = EXPORT_OUTPUT_DIR_PATH+EXPORT_OUTPUT_NAME

import torch
import dataset
import os
import time
import math
from torchsummary import summary
from model import DSCNN
from utils import remove_txt, parameter_generation
from copy import deepcopy
from pthflops import count_ops
import csv
from train import Train
import shutil

# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(torch.version.__version__)
print(device)

# Model generation
model = DSCNN(use_bias=True)
model.to(device)


if STEP_DO_TRAIN:
    
    # Parameter generation
    training_parameters, data_processing_parameters = parameter_generation()  # To be parametrized

    # Dataset generation
    audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    print("Dataset split (Train/valid/test): " + str(train_size) + "/" + str(valid_size) + "/" + str(test_size))

    # Model analysis
    print("Printing model summary...")
    summary(model, (1, 49, data_processing_parameters['feature_bin_count']))
    dummy_input = torch.rand(1, 1, 49, data_processing_parameters['feature_bin_count']).to(device)
    count_ops(model, dummy_input)
    # Training initialization
    print("Initializing training environment...")
    trainining_environment = Train(audio_processor, training_parameters, model, device)

    # Removing stored inputs and activations
    print("Removing stored inputs and activations...")
    remove_txt()

    # Start training
    print("Starting training...")
    start = time.clock_gettime(0)
    trainining_environment.train(model)
    print('Finished Training on GPU in {:.2f} seconds'.format(time.clock_gettime(0) - start))

if STEP_DO_EXPORT_MODEL:
    # Load pretrained weights
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    
    # first_layer_weights = model.conv1.weight

    # 
    # print("Shape of first layer weights:", first_layer_weights.shape)

    # 
    # print("First kernel weights (first filter) in the first layer:")
    # print(first_layer_weights[1].detach().cpu().numpy())  

    

    path = os.path.join(".", EXPORT_OUTPUT_DIR_PATH)

    # Remove the specified
    # file path
    if os.path.isdir(EXPORT_OUTPUT_DIR_PATH):
        print("")
    else:
        os.mkdir(path)


    try:
        with open(EXPORT_OUTPUT_PATH, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            
            writer.writerow(["Name", "Shape", "Values"])

            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_name = name
                    param_shape = list(param.shape)
                    param_values = param.detach().cpu().numpy().flatten().tolist()
                    
                    # Write parameter details to CSV
                    writer.writerow([param_name, param_shape, param_values])

        print(f"Model parameters exported successfully to {EXPORT_OUTPUT_PATH}.")
    except Exception as e:
        print(f"Error exporting model parameters: {e}")