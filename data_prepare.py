from PIL import Image
import torch
from torchvision import transforms

# Image path
image_path = "Vehicles/Cars/Car (1).jpg"

# Open image
image = Image.open(image_path).convert("RGB")

print("Original image size:")
print(image.size)

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Convert image to tensor
image_tensor = transform(image)

print("\nTensor shape:")
print(image_tensor.shape)

print("\nTensor datatype:")
print(image_tensor.dtype)

print("\nMinimum pixel value:")
print(image_tensor.min())

print("\nMaximum pixel value:")
print(image_tensor.max())