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

- streamlit
- Pillow
- numpy
- opencv-python
- pandas
- hnswlib
- facenet-pytorch
- torch
- torchvision
- mediapipe

## Installation
- Clone the repository: git clone <https://github.com/rezabrati/celebrity.git>
- Create a virtual environment (optional but recommended):
- Install the required packages: pip install -r requirements.txt
 
## Usage
- Run the Streamlit app: streamlit run main.py
- Open your web browser and go to: http://localhost:8501
- Upload an image:
- After processing, the app will display your photo with facial landmarks and the top 3 celebrity matches.