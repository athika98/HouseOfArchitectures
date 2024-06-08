# scripts/check_file.py
import os

# Absolute path to the data directory
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')

# Walk through all subdirectories and files in the data directory
for root, dirs, files in os.walk(data_dir):
    for file in files:
        # Construct the full file path
        file_path = os.path.join(root, file)
        try:
            # Try to open the file in binary read mode
            with open(file_path, 'rb') as f:
                pass  # If successful, do nothing and close the file
        except Exception as e:
            # If an error occurs, print the file path and the error message
            print(f"Error with file {file_path}: {e}")
