# Project Readme

## Project Structure

This project contains the following folders and files:

- `Dataset/`: Contains two subfolders, `Training/` and `Testing/`, each of which contains multiple subfolders with class labels, and each of these folders contains images for the corresponding class.

- `Preprocessing and Data Cleaning/`: Contains four Python files for data cleaning:
    - `make_it_gray.py`: Converts all images to grayscale.
    - `brightness_adjustments.py`: Adjusts Brightness of the Images
    - `image_attributes.py`: Gives insights about the attributes of the images
    - `resize-focused.py`: It will resize images for the focused class

- `Data Visualization/`: Contains a single Python file for data visualization:
    - `data_visualization.py`: Performs three visualization tasks - class distribution, sample images, and pixel intensity distribution.

## Data Cleaning

To perform data cleaning, follow these steps:

1. Navigate to the `Preprocessing and Data Cleaning/` folder.
2. Run the Python script `make_it_gray.py` to convert all images to grayscale.
3. Run the other data cleaning scripts if needed.

There are no special instructions for running these scripts. Simply execute them to perform the specified data cleaning tasks.

## Data Visualization

To visualize the data, follow these steps:

1. Navigate to the `Data Visualization/` folder.
2. Run the Python script `visualize_data.py`.
3. The script will generate visualizations for class distribution, sample images, and pixel intensity distribution.

Again, there are no special instructions for running the data visualization script. Execute it, and it will generate the visualizations as described.

That's it! You can now clean your data and visualize it using the provided scripts.
