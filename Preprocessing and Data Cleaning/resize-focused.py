import os
from PIL import Image



def resize_images(directory_path, output_directory, new_width=48, new_height=48):
    """
    Resize all images in a directory to the specified width and height.

    Args:
    - directory_path (str): Path to the directory containing the images.
    - output_directory (str): Path to the directory where resized images will be saved.
    - new_width (int): Width of the resized images.
    - new_height (int): Height of the resized images.
    """

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List all files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            with Image.open(os.path.join(directory_path, file_name)) as img:
                # Resize image
                img_resized = img.resize((48, 48))

                # Save resized image to the output directory
                img_resized.save(os.path.join(output_directory, file_name))


directory_path = 'C:\AI_Project\A.I.ducation_Analytics\TrainingDataSet\Focused'  # Replace with the path to your folder
output_directory = 'C:\AI_Project\A.I.ducation_Analytics\TrainingDataSet\Focused'  # Replace with the path where you want to save resized images
resize_images(directory_path, output_directory)
