import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import numpy as np

class_labels = ["Neutral", "Engaged/Focused", "Bored/Tired", "Angry/Irritated"]
class_distribution = [595, 534, 500, 534]  # Images in each class

# Paths to the directories containing images for each class
class_directories = {
    "Neutral": "../Dataset/Training/Neutral",
    "Engaged/Focused": "../Dataset/Training/Focused",
    "Bored/Tired": "../Dataset/Training/Bored",
    "Angry/Irritated": "../Dataset/Training/Angry"
}


# Plot Class distribution

plt.figure(figsize=(8, 4))
plt.bar(class_labels, class_distribution)
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()


# Sample Images

# Initialize a 5x5 grid for displaying the images
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

# Randomly select 25 images from the classes
selected_images = []

# Create a list to track which classes have been represented
remaining_classes = list(class_directories.keys())

while len(selected_images) < 25:
    # Shuffle the list of remaining classes to randomize selection
    random.shuffle(remaining_classes)

    for class_label in remaining_classes:
        dir_path = class_directories[class_label]
        image_files = os.listdir(dir_path)

        if len(image_files) > 0:
            random_image = Image.open(os.path.join(dir_path, random.choice(image_files)))
            selected_images.append((random_image, class_label))

# Display the selected images in the grid
for i in range(25):
    img, class_label = selected_images[i]
    axes[i // 5, i % 5].imshow(img)
    axes[i // 5, i % 5].set_title(class_label)
    axes[i // 5, i % 5].axis('off')

plt.suptitle("Randomly Selected Images from Different Classes")
plt.tight_layout()
plt.show()


#Pixel Intensity Distribution

# Initialize an empty list to store pixel intensity values
pixel_intensities = []

# Collect pixel intensity values from the selected images
for img, _ in selected_images:
    # Convert the image to grayscale
    grayscale_img = img.convert("L")
    pixel_intensity = np.array(grayscale_img).ravel()
    pixel_intensities.extend(pixel_intensity)

# Plot the pixel intensity distribution for grayscale images
plt.figure(figsize=(8, 6))
plt.hist(pixel_intensities, bins=256, range=(0, 256), color='gray', alpha=0.7)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Pixel Intensity Distribution for Selected Images")
plt.grid()
plt.show()