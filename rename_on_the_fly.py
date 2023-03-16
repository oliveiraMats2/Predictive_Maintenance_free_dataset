import os

# Set the directory path containing the files you want to rename
dir_path = "/path/to/directory/"

# Iterate through each file in the directory
for filename in os.listdir(dir_path):
    if filename.startswith("mse_epoch_") and filename.endswith(".png"):
        # Extract the numeric part of the filename
        num_str = filename.replace("mse_epoch_", "").replace(".png", "")
        # Rename the file
        os.rename(os.path.join(dir_path, filename), os.path.join(dir_path, f"{num_str}.png"))