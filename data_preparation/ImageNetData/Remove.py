import os

def remove_prefix_zeros_from_folders(base_path):
    # Get a list of all directories in the base path
    folders = [f for f in os.scandir(base_path) if f.is_dir()]

    for folder in folders:
        old_name = folder.name
        new_name = old_name.lstrip('0')  # Remove leading zeros
        if new_name != old_name:
            old_path = os.path.join(base_path, old_name)
            new_path = os.path.join(base_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed folder: '{old_name}' to '{new_name}'")

# Specify the base path (current directory)
base_path = os.getcwd()
remove_prefix_zeros_from_folders(base_path)
