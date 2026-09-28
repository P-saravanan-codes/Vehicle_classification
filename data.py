import kagglehub

# Download latest version
path = kagglehub.dataset_download("mohamedmaher5/vehicle-classification")

print("Path to dataset files:", path)