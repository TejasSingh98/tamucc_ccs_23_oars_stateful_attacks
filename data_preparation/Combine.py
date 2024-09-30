import os
from PIL import Image

def rename_images(base_path):
    for root, _, files in os.walk(base_path):
        label = os.path.basename(root)
        image_counter = 1
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                old_path = os.path.join(root, file)
                new_name = f"image_{image_counter}_label_{label}{os.path.splitext(file)[1]}"
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                image_counter += 1

def combine_images(base_path):
    images = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                images.append(Image.open(os.path.join(root, file)))
    
    if not images:
        print("No images found.")
        return

    # Calculate the dimensions of the combined image
    widths, heights = zip(*(img.size for img in images))
    total_width = max(widths)
    total_height = sum(heights)

    # Create a new blank image with the total width and height
    combined_image = Image.new('RGB', (total_width, total_height))

    # Paste each image into the combined image
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height

    # Save the combined image
    combined_image.save("combined_image.jpg")
    print("Combined image saved as combined_image.jpg")

def rename_and_combine_images(base_path):
    rename_images(base_path)
    combine_images(base_path)

# Specify the base path (current directory)
base_path = os.getcwd()
rename_and_combine_images(base_path)
