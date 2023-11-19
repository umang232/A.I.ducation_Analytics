import torch
from torchvision import transforms
from PIL import Image
from utils.dataloader import get_dataloader

# Define the path to the saved model
model_path = "../saved_models"

# Load the saved model
model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_path):
    # Load and preprocess the input image
    img = Image.open(image_path)
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension

    # Make predictions
    with torch.no_grad():
        output = model(img)

    # Assuming your model predicts class probabilities, get the predicted class
    _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

# Example usage for an individual image
image_path = "individual_image"
predicted_class = predict_image(image_path)
print(f"The model predicts the image belongs to class {predicted_class}")

# Example usage for a complete dataset
# Assuming you have a DataLoader named 'data_loader' for your dataset
for images, _ in data_loader:
    with torch.no_grad():
        outputs = model(images)

    # Process the model outputs as needed for your application
    # (e.g., storing predictions, performing further analysis, etc.)
