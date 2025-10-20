import os
import kaggle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_dataset():
    """Downloads and unzips the fashion dataset from Kaggle if not present."""
    dataset_slug = 'nguyngiabol/colorful-fashion-dataset-for-object-detection'
    output_path = './data/raw'

    if os.path.exists(output_path):
        logging.info(f"Dataset already found at '{output_path}'. Skipping download.")
        return
    
    logging.info(f"Dataset not found. Downloading '{dataset_slug}' from Kaggle...")
    os.makedirs(output_path, exist_ok=True)
    try:
        kaggle.api.dataset_download_files(dataset_slug, path=output_path, unzip=True)
        logging.info("Download and extraction complete.")
    except Exception as e:
        logging.error(f"Failed to download dataset. Ensure your kaggle.json is set up. Error: {e}")

if __name__ == '__main__':
    download_dataset()