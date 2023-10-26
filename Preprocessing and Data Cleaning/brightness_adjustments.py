import os
from PIL import Image, ImageEnhance

def adjust_image_lighting(img_path, output_path, brightness_factor=1.0, contrast_factor=1.0, color_balance_factor=1.0):
    """
    Adjust the lighting conditions of an image.

    Args:
    - img_path (str): Path to the source image.
    - output_path (str): Path to save the adjusted image.
    - brightness_factor (float): Factor to adjust the brightness. >1 to increase, <1 to decrease.
    - contrast_factor (float): Factor to adjust the contrast. >1 to increase, <1 to decrease.
    - color_balance_factor (float): Factor to adjust the color balance. >1 to make it cooler, <1 to make it warmer.
    """

    with Image.open(img_path) as img:
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)

        # Adjust contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        # Adjust color balance (for simplicity, we'll adjust the red channel to influence warmth)
        r, g, b = img.split()
        enhancer = ImageEnhance.Brightness(r)
        r = enhancer.enhance(color_balance_factor)
        img = Image.merge("RGB", (r, g, b))

        # Save adjusted image
        img.save(output_path)

def adjust_images_in_directory(directory_path, output_directory, brightness_factor=1.0, contrast_factor=1.0, color_balance_factor=1.0):
    """Adjust the lighting conditions of all images in a directory."""
    for file_name in os.listdir(directory_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            src_path = os.path.join(directory_path, file_name)
            dest_path = os.path.join(output_directory, file_name)
            adjust_image_lighting(src_path, dest_path, brightness_factor, contrast_factor, color_balance_factor)

# Example usage:
directory_path = 'C:\AI_Project\A.I.ducation_Analytics\TrainingDataSet'    # Replace with the path to your images
output_directory = 'C:\AI_Project\A.I.ducation_Analytics\TrainingDataSet'      # Replace with where you want to save adjusted images
adjust_images_in_directory(directory_path, output_directory, 1.1, 1.1, 0.9)
