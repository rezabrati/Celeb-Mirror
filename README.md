# Celeb Mirror: Who Do You Resemble?

## Overview

**Celeb Mirror** is a web application built with Streamlit that allows users to upload their photos and discover which celebrities they resemble the most. Using advanced facial recognition technology, the app analyzes the uploaded image and retrieves similar images from a celebrity database.

## Features

- Upload an image in various formats (JPEG, JPG, PNG).
- View your uploaded photo processed with facial landmarks.
- Get the top 3 celebrity look-alikes based on the uploaded photo.
- Intuitive user interface for a seamless experience.

## Requirements

Make sure you have the following libraries installed in your environment. You can install them using `pip`:
```plaintext
streamlit
Pillow
numpy
opencv-python
pandas
hnswlib
facenet-pytorch
torch
torchvision
mediapipe
```
## Installation
- Clone the repository: 
```bash
git clone <https://github.com/rezabrati/celebrity.git>
```
- Create a virtual environment (optional but recommended):
- Install the required packages: 
```bash
pip install -r requirements.txt
```

## Generate Celebrity Embeddings

Before running the app, you need to create a database of celebrity images and their corresponding embeddings. Follow these steps:

Download celebrity images and create embeddings: The script will download images for a given list of celebrities and compute their facial embeddings using the FaceNet model.

Example usage:
```python
from utils import EmbeddingSaver, FaceNetModel

# Define your base directory for storing images
base_directory = 'img'

# Initialize the FaceNet model
facenet_model = FaceNetModel()

# Initialize EmbeddingSaver with the base directory and model
embedding_saver = EmbeddingSaver(base_dir=base_directory, facenet_model=facenet_model)

# Provide search keywords for downloading images
search_keywords = ["Scarlett Johansson", "Tom Hanks", "Adele"]

# Run the embedding generation process
embedding_saver.run(search_keywords=search_keywords, num_images=2)

```
## Usage
- Run the Streamlit app: 
```bash
streamlit run main.py
```
- Open your web browser and go to: http://localhost:8501
- Upload an image:
- After processing, the app will display your photo with facial landmarks and the top 3 celebrity matches.