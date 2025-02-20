
import torch

def load_model_and_print_details(pth_file):
    try:
        # Load the .pth file
        model_data = torch.load(pth_file, map_location=torch.device('cpu'))

        # If the file contains a model's state_dict
        if isinstance(model_data, dict):
            print("Keys in the loaded file:")
            for key in model_data.keys():
                print(f"- {key}")

            if 'state_dict' in model_data:
                state_dict = model_data['state_dict']
                print("\nModel Parameters:")
                for name, param in state_dict.items():
                    print(f"Parameter Name: {name}")
                    print(f"Shape: {param.shape}")
                    print(f"Values: {param}")
                    print("-" * 40)

            # Print non-parameter data
            print("\nNon-parameter data:")
            for key, value in model_data.items():
                if key != 'state_dict':
                    print(f"Key: {key}")
                    print(f"Value: {value}")
                    print("-" * 40)
        else:
            print("The loaded file does not contain a dictionary.")

    except Exception as e:
        print(f"Error loading the model: {e}")

# Replace 'your_model.pth' with the path to your .pth file
pth_file = 'your_model.pth'
load_model_and_print_details(pth_file)

