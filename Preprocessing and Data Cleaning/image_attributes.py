import os
from PIL import Image
def get_image_attributes(directory_path):
    """
    Extract image attributes for all images in a given directory.

    Args:
    - directory_path (str): Path to the directory containing the images.

    Returns:
    - dict: Dictionary containing image attributes.
    """
    image_attributes = {
        'width': set(),
        'height': set(),
        'mode': set(),
        'resolution': set()
    }

    # List all files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            with Image.open(os.path.join(directory_path, file_name)) as img:
                width, height = img.size
                image_attributes['width'].add(width)
                image_attributes['height'].add(height)
                image_attributes['mode'].add(img.mode)
                image_attributes['resolution'].add(img.info.get('dpi', (None, None)))

    return image_attributes


directory_path = 'C:\AI_Project\A.I.ducation_Analytics\TrainingDataSet\Angry'  # Replace with the path to your folder
attributes = get_image_attributes(directory_path)

print("Widths:", attributes['width'])
print("Heights:", attributes['height'])
print("Modes:", attributes['mode'])
print("Resolutions:", attributes['resolution'])