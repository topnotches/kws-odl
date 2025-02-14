import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

# Get a list of all CSV files in the directory
csv_files = glob.glob("./val_loss_logs/*.csv")

# Check if any files were found
if not csv_files:
    print("No CSV files found in the directory.")
    exit()

# Create a figure for individual plots
plt.figure(figsize=(10, 6))

# Dictionary to store all runs
all_data = {}

# Iterate over all files and plot their data
for file in csv_files:
    df = pd.read_csv(file)
    
    # Store data for averaging
    all_data[file] = df[["Epoch", "Val_Acc"]]
    
    # Plot individual runs
    plt.plot(df["Epoch"], df["Val_Acc"], alpha=0.5)  # Use filename as label

# Set labels and title for individual runs
plt.xlabel("Epoch")
plt.ylabel("Val_Acc")
plt.title("Validation Accuracy vs. Epoch (All Runs)")

# Set y-axis limits dynamically
all_vals = pd.concat([df["Val_Acc"] for df in all_data.values()])
plt.ylim(all_vals.min() - 0.01, all_vals.max() + 0.01)

# Rotate x-tick labels for better visibility
plt.xticks(rotation=45, ha='right')

# Show legend
plt.legend()

# Adjust layout to prevent label clipping
plt.tight_layout()

# Save the individual runs plot
plt.savefig("val_acc_all.png")

# Close figure to free resources
plt.close()

print("Plot saved as 'val_acc_all.png'")

### Compute and plot the average accuracy

# Merge all data on "Epoch"
merged_df = pd.concat(all_data.values())

# Compute the mean for each epoch
average_df = merged_df.groupby("Epoch").mean().reset_index()

# Create a figure for the average plot
plt.figure(figsize=(10, 6))

# Plot the average validation accuracy
plt.plot(average_df["Epoch"], average_df["Val_Acc"], color="red", linewidth=2, label="Average Val_Acc")

# Set labels and title for average plot
plt.xlabel("Epoch")
plt.ylabel("Val_Acc")
plt.title("Average Validation Accuracy vs. Epoch")

# Set y-axis limits dynamically
plt.ylim(average_df["Val_Acc"].min() - 0.01, average_df["Val_Acc"].max() + 0.01)

# Rotate x-tick labels for better visibility
plt.xticks(rotation=45, ha='right')

# Show legend
plt.legend()

# Adjust layout
plt.tight_layout()

# Save the average plot
plt.savefig("val_acc_avg.png")

# Close figure to free resources
plt.close()

print("Plot saved as 'val_acc_avg.png'")

# Check if any files were found
if not csv_files:
    print("No CSV files found in the directory.")
    exit()

# Create a figure for individual plots
plt.figure(figsize=(10, 6))

# Dictionary to store all runs
all_data = {}

# Iterate over all files and plot their data
for file in csv_files:
    df = pd.read_csv(file)
    
    # Store data for averaging
    all_data[file] = df[["Epoch", "Val_Loss"]]
    
    # Plot individual runs
    plt.plot(df["Epoch"], df["Val_Loss"], alpha=0.5)  # Use filename as label

# Set labels and title for individual runs
plt.xlabel("Epoch")
plt.ylabel("Val_Loss")
plt.title("Validation Loss vs. Epoch (All Runs)")

# Set y-axis limits dynamically
all_vals = pd.concat([df["Val_Loss"] for df in all_data.values()])
plt.ylim(all_vals.min() - 0.01, all_vals.max() + 0.01)

# Rotate x-tick labels for better visibility
plt.xticks(rotation=45, ha='right')

# Show legend
plt.legend()

# Adjust layout to prevent label clipping
plt.tight_layout()

# Save the individual runs plot
plt.savefig("Val_Loss_all.png")

# Close figure to free resources
plt.close()

print("Plot saved as 'Val_Loss_all.png'")

### Compute and plot the average accuracy

# Merge all data on "Epoch"
merged_df = pd.concat(all_data.values())

# Compute the mean for each epoch
average_df = merged_df.groupby("Epoch").mean().reset_index()

# Create a figure for the average plot
plt.figure(figsize=(10, 6))

# Plot the average validation Loss
plt.plot(average_df["Epoch"], average_df["Val_Loss"], color="red", linewidth=2, label="Average Val_Loss")

# Set labels and title for average plot
plt.xlabel("Epoch")
plt.ylabel("Val_Loss")
plt.title("Average Validation Loss vs. Epoch")

# Set y-axis limits dynamically
plt.ylim(average_df["Val_Loss"].min() - 0.01, average_df["Val_Loss"].max() + 0.01)

# Rotate x-tick labels for better visibility
plt.xticks(rotation=45, ha='right')

# Show legend
plt.legend()

# Adjust layout
plt.tight_layout()

# Save the average plot
plt.savefig("Val_Loss_avg.png")

# Close figure to free resources
plt.close()

print("Plot saved as 'Val_Loss_avg.png'")
