import os
from PIL import Image


def convert_to_gray(directory_path, output_directory):
    """
    Convert all images in a directory to grayscale.

    Args:
    - directory_path (str): Path to the directory containing the images.
    - output_directory (str): Path to the directory where grayscale images will be saved.
    """

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List all files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            with Image.open(os.path.join(directory_path, file_name)) as img:
                # Convert image to grayscale
                img_gray = img.convert('L')

                # Save grayscale image to the output directory
                img_gray.save(os.path.join(output_directory, file_name))


directory_path = 'C:\AI_Project\A.I.ducation_Analytics\TrainingDataSet\Focused'  # Replace with the path to your folder
output_directory = 'C:\AI_Project\A.I.ducation_Analytics\TrainingDataSet\Focused'  # Replace with the path where you want to save grayscale images
convert_to_gray(directory_path, output_directory)
