import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("MYERRORS.csv")
# Print the loaded data
print(df)

# Plot the error values
plt.plot(df["Index"], df["Error"], label='Validation Error')

# Set labels and title
plt.xlabel("Index")
plt.ylabel("Error")
plt.title("Validation Errors")

# Set y-axis limits for better visibility (example limits)
plt.ylim(df["Error"].min() - 0.01, df["Error"].max() + 0.01)  # Adjust as needed

# Rotate x-tick labels for better visibility
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent label clipping
plt.tight_layout()

# Save the plot as an image
plt.savefig("validation_errors_plot.png")

# Optionally, close the plot to free resources
plt.close()

print("Plot saved as 'validation_errors_plot.png'")
