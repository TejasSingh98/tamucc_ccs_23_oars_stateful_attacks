from PIL import Image
import os

# Directory containing the images
directory = 'imgs'

# Output directory for resized images
output_directory = 'imgs1'

# Ensure the output directory exists, create if not
os.makedirs(output_directory, exist_ok=True)

# Target dimensions
target_size = (224, 224)

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith('.png')]

# Iterate through each image file
for image_name in image_files:
    # Open the image
    image_path = os.path.join(directory, image_name)
    img = Image.open(image_path)
    
    # Resize the image to target size (using ANTIALIAS to maintain quality)
    img_resized = img.resize(target_size, Image.ANTIALIAS)
    
    # Transpose dimensions to (3, 224, 224) assuming RGB mode
    img_transposed = img_resized.transpose(Image.TRANSPOSE)
    
    # Save the transposed image
    output_path = os.path.join(output_directory, image_name)
    img_transposed.save(output_path)
    
    print(f'Transposed and saved {image_name} to {output_path}')

print('All images transposed and saved successfully.')

