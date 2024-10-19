import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from bing_image_downloader import downloader
import sys
sys.path.append('../')
from utils import FaceNetModel


class EmbeddingSaver:
    def __init__(self, base_dir, facenet_model, output_csv='embeddings_celeb3.csv'):
        self.base_dir = base_dir
        self.facenet_model = facenet_model
        self.output_csv = output_csv
        self.image_names = []
        self.all_embeddings = []

    def get_facenet_embedding(self, image_path):
        """Method to get the embedding for an image using FaceNet model."""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.facenet_model.preprocess(image).unsqueeze(0)  # Assuming a method in facenet_model for preprocessing
        with torch.no_grad():
            embedding = self.facenet_model.model(image_tensor)
        return embedding.squeeze().numpy()

    def download_images(self, search_keywords, num_images=5):
        """Download portrait images based on search keywords."""
        for keyword in search_keywords:
            try:
                # Modify the search term to focus on portrait images
                search_term = f"{keyword} portrait"
                print(f"Downloading portrait images for: {keyword}")

                # Download images using bing_image_downloader
                downloader.download(
                                    search_term, 
                                    limit=num_images, 
                                    output_dir=self.base_dir,
                                    adult_filter_off=True, 
                                    force_replace=False, 
                                    timeout=60
                                    )
                print(f"Downloaded portrait images for: {keyword}")
            except Exception as e:
                print(f"Error downloading {keyword}: {e}")

    def process_directory(self):
        """Iterate over directories and images to collect embeddings."""
        for directory in os.listdir(self.base_dir):
            print(f"Processing directory: {directory}")
            directory_path = os.path.join(self.base_dir, directory)

            # Iterate through each image file in the directory
            for file in os.listdir(directory_path):
                if file.endswith('.jpg'):
                    image_path = os.path.join(directory_path, file)  # Full image path
                    embedding = self.get_facenet_embedding(image_path)  # Get the embedding
                    print(f"Stored embedding for {file}")

                    # Append image name and embedding to lists
                    self.image_names.append(image_path)
                    self.all_embeddings.append(embedding)

    def save_embeddings(self):
        """Convert embeddings and image names to a DataFrame and save to CSV."""
        df = pd.DataFrame(self.all_embeddings)  # Convert embeddings to DataFrame
        df['image_name'] = self.image_names  # Add image names as a column
        df.to_csv(self.output_csv, index=False)  # Save to CSV

    def run(self, search_keywords=None, num_images=5):
        """Main method to download images, process them, and save embeddings."""
        if search_keywords:
            self.download_images(search_keywords, num_images)  # Download images first if keywords are provided
        self.process_directory()  # Process the images for embeddings
        self.all_embeddings = np.array(self.all_embeddings)  # Convert list of embeddings to 2D array
        self.save_embeddings()  # Save embeddings to CSV

# Example usage:
# Assuming `facenet_model` is an instance of the FaceNetModel class defined previously
base_directory = 'img'
embedding_saver = EmbeddingSaver(base_dir=base_directory, facenet_model= FaceNetModel())

# Provide search keywords for downloading images
search_keywords = ["Scarlett Johansson", "Tom Hanks", "Adele"]
embedding_saver.run(search_keywords=search_keywords, num_images=2)
