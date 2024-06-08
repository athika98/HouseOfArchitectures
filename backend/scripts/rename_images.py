# scripts/rename_images.py

import os

# Directory containing the images
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')

# Walk through all subdirectories
for root, dirs, files in os.walk(base_dir):
    # Initialize a counter for naming the images within each directory
    counter = 1
    
    for file_name in files:
        # Full old file path
        old_file = os.path.join(root, file_name)
        
        # Skip directories, only process files
        if os.path.isfile(old_file):
            # Check for image file extensions (case insensitive)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                # Extract the folder name and replace spaces with underscores
                folder_name = os.path.basename(root).replace(' ', '_')
                
                # Generate new file name with folder name included
                new_name = f'image_{folder_name}_{counter:02d}.jpg'
                
                # Full new file path
                new_file = os.path.join(root, new_name)
                
                # Ensure the new file name is unique by adding a counter if needed
                while os.path.exists(new_file):
                    counter += 1
                    new_name = f'image_{folder_name}_{counter:02d}.jpg'
                    new_file = os.path.join(root, new_name)

                # Rename the file
                os.rename(old_file, new_file)
                
                # Increment the counter for the next file
                counter += 1
