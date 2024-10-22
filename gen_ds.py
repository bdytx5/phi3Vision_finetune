import os
import pandas as pd
from datasets import load_dataset
import requests
from PIL import Image
from io import BytesIO

# Function to download an image from a URL and save it locally
def download_image(image_url, save_path):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Check if the request was successful
        image = Image.open(BytesIO(response.content))
        image.save(save_path)
        return True
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")
        return False

# Download the dataset from Hugging Face
dataset = load_dataset('DBQ/Burberry.Product.prices.United.States')

# Convert the Hugging Face dataset to a Pandas DataFrame
df = dataset['train'].to_pandas()

# Create directories to save the dataset and images
dataset_dir = './data/burberry_dataset'
images_dir = os.path.join(dataset_dir, 'images')
os.makedirs(images_dir, exist_ok=True)

# Filter out rows where image download fails
filtered_rows = []
for idx, row in df.iterrows():
    image_url = row['imageurl']
    image_name = f"{row['product_code']}.jpg"
    image_path = os.path.join(images_dir, image_name)
    if download_image(image_url, image_path):
        row['local_image_path'] = image_path
        filtered_rows.append(row)

# Create a new DataFrame with the filtered rows
filtered_df = pd.DataFrame(filtered_rows)

# Save the updated dataset to disk
dataset_path = os.path.join(dataset_dir, 'burberry_dataset.csv')
filtered_df.to_csv(dataset_path, index=False)

print(f"Dataset and images saved to {dataset_dir}")
