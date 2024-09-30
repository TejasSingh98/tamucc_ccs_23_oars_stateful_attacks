import os
import shutil
 
def rename_and_combine_images(destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    # Get the current working directory
    current_dir = os.getcwd()
    # List all items (files and directories) in the current directory
    items = os.listdir(current_dir)
    # Initialize the image counter globally
    image_counter = 0 
    # Iterate through each item
    for item in items:
        item_path = os.path.join(current_dir, item)
        # Check if the item is a directory
        if os.path.isdir(item_path):
            try:
                files = os.listdir(item_path)
                # Iterate through files in the directory
                for file in files:
                    # Check if the file is an image
                    if file.endswith(('.jpg', '.jpeg', '.png', '.JPEG')):
                        label = item  # Use the directory name as the label
                        file_extension = os.path.splitext(file)[1]  # Get the file extension
                        new_file_name = f"image_{image_counter}_label_{label}{file_extension}"
                        src_file_path = os.path.join(item_path, file)
                        dst_file_path = os.path.join(destination_folder, new_file_name)
                        # Copy and rename the file
                        shutil.copy(src_file_path, dst_file_path)
                        print(f"Copied and renamed {file} to {new_file_name}")
                        # Increment the image counter globally
                        image_counter += 1
            except Exception as e:
                print(f"Error accessing files in {item}: {e}")
 
# Define the destination folder variable
destination_folder = '/home/ubuntu/development/data/Imagenet'  # Destination folder to save combined images
 
# Call the function to rename and combine images from all folders
rename_and_combine_images(destination_folder)
