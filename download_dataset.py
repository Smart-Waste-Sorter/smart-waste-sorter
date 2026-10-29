from huggingface_hub import snapshot_download
import os
import shutil

# Hugging Face dataset
REPO_ID = "bryandts/waste_organic_anorganic_classification"

# Download the dataset files
print("Downloading dataset...")

dataset_path = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset"
)

print("Dataset downloaded to:")
print(dataset_path)

# Show the files
print("\nDataset files:")

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        print(os.path.join(root, file))