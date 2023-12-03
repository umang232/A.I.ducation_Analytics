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

## Experiment


This folder contains the main experiment scripts for training and evaluating the CNN models on the dataset. Each script corresponds to a different experiment:

1. experiment1.py: Trains and evaluates the base CNN model architecture.
2. experiment2.py: Variant 1 - Alters the number of convolutional layers to study the impact on model performance.
3. experiment3.py: Variant 2 - Adjusts kernel sizes in convolutional layers to analyze trade-offs in feature recognition and computational cost.

## Models

This folder contains the definitions of the CNN models used in the experiments:

1. cnn_model.py: Defines the base CNN class along with two variant classes. The base class is a convolutional neural network designed for image classification. Variant1CNN and Variant2CNN are modified versions of the base architecture, exploring different aspects of CNN design like layer depth and kernel size.

## Utils

This folder contains utility scripts that support model training, evaluation, and data handling:

1. dataloader.py: Provides functions to load and preprocess the dataset, creating data loaders for training, validation, and testing.
2. train_eval.py: Contains functions for training models and evaluating their performance on the dataset. Includes implementation of early stopping for training optimization.
3. evaluation_matrix.py: (Note: Mentioned in the chat but not detailed) Presumably contains functions for calculating and reporting various evaluation metrics for the models.
