import os
import json

def create_json_from_images(directory, output_file):
    # Dictionary to store image paths and their corresponding labels
    image_dict = {}

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if (filename.endswith(".png") or filename.endswith(".jpg")):  # Ensure it's a PNG file
            # Construct the full path to the file
            full_path = os.path.join("imagenet-data", filename)
            
            # Parse the label from the filename
            # Assuming the format is image_X_label_Y.png and you want Y
            label = int(filename.split('_')[-1].split('.')[0])
            
            # Add to dictionary
            image_dict[full_path] = label

    # Write the dictionary to a JSON file
    with open(output_file, 'w') as f:
        json.dump(image_dict, f, indent=4)

    print(f"JSON file '{output_file}' created with {len(image_dict)} entries.")

# Usage
directory_path = 'imagenet-data/'  # Path to the directory containing the images
output_json_file = 'imagenet-data.json'        # Path where the JSON file will be saved
create_json_from_images(directory_path, output_json_file)

