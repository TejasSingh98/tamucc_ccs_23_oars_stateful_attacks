import os

def create_pseudo_json(directory, output_file):
    # Start building a 'pseudo-JSON' string
    pseudo_json = "{\n"

    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    # Process each file in the directory
    for filename in os.listdir(directory):
        if (filename.endswith(".png") or filename.endswith(".jpg")):  # Filter to include only PNG files
            # Extract the label from the filename
            label = filename.split('_')[-1].split('.')[0]

            # Construct the full path to be stored in the 'JSON'
            full_path = f"    \"{label}\": [\n        \"imgs/{filename}\"\n    ],\n"

            # Append this entry to the pseudo-JSON string
            pseudo_json += full_path

    # Close the structure
    pseudo_json = pseudo_json.rstrip(',\n') + "\n}"

    # Write the 'pseudo-JSON' string to a file
    with open(output_file, 'w') as f:
        f.write(pseudo_json)

    print(f"File '{output_file}' created with duplicate keys.")

# Directory containing the images
directory_path = 'imagenet-data'  # Change to the path where your images are stored

# Output 'pseudo-JSON' file name
output_json_file = 'imagenet-data2.json'

# Create the 'pseudo-JSON' file
create_pseudo_json(directory_path, output_json_file)

