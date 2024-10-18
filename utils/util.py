import matplotlib.pyplot as plt
from PIL import Image
import mediapipe as mp
import numpy as np
import os
import cv2
import pandas as pd
import hnswlib
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms

# Define your local base path where the images are stored
BASE_PATH = "C:/Users/vcc/Desktop/celeb-face"

# Class for path modification and showing similar images
class ImageUtils:
    def __init__(self, base_path=BASE_PATH):
        self.base_path = base_path

    def modify_paths(self, image_paths):
        """ Modify the paths to point to the local directory """
        local_paths = []
        for path in image_paths:
            # Remove the Google Drive path part and replace with the local base path
            local_path = path.replace("/content/drive/MyDrive/", self.base_path + "/")
            local_paths.append(local_path)
        return local_paths

    def show_similar_images(self, similar_images, image_names, title_names, rows=2, cols=3):
        """ Display top similar images in a 2x3 grid """
        fig, axes = plt.subplots(rows, cols, figsize=(12, 8))  # Create a figure with 2 rows and 3 columns
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i, (img_path, ax, title) in enumerate(zip(similar_images, axes, title_names)):
            img = Image.open(img_path)  # Open the image file
            ax.imshow(img)  # Display the image
            ax.set_title(title)  # Set the title (celebrity name)
            ax.axis('off')  # Hide the axis (optional)

        # Hide any remaining empty axes if there are less than 6 images
        for ax in axes[len(similar_images):]:
            ax.axis('off')

        plt.tight_layout()  # Adjust the spacing between images
        plt.show()

# Class for image processing (applying face mesh)
class FaceMeshProcessor:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        # Define the silver color for drawing the mesh
        self.silver_color = (220, 220, 192)
        self.custom_drawing_spec = self.mp_drawing.DrawingSpec(color=self.silver_color, thickness=1)

    def process_image(self, image):
        """ Apply the face mesh to the uploaded image """
        # Convert the image to RGB
        rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        
        # Process the image with face mesh
        result = self.face_mesh.process(rgb_image)
        
        # Convert back to BGR for OpenCV
        image_with_mesh = np.array(image)
        
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Draw the face mesh on the image
                self.mp_drawing.draw_landmarks(
                    image=image_with_mesh,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,  # Draw full face mesh
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.custom_drawing_spec  # Silver mesh
                )
        return image_with_mesh

# Class for handling model setup and embeddings
class FaceNetModel:
    def __init__(self):
        """Initialize the FaceNet model."""
        self.model = InceptionResnetV1(pretrained='vggface2').eval()  # Use the VGGFace2 pretrained weights
        # Define the preprocessing pipeline for input images
        self.preprocess = transforms.Compose([
            transforms.Resize(160),
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_facenet_embedding(self, image_path):
        """Generate embedding using FaceNet for a given image path."""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            embedding = self.model(image_tensor)
        return embedding.squeeze().numpy()

# Class for finding similar images using embeddings
class ImageFinder:
    def __init__(self, embedding_file="embeddings_celeb150_portrait.csv"):
        # Load embeddings from a CSV file
        self.embedding_file = embedding_file
        self.df = pd.read_csv(self.embedding_file)
        self.image_names = self.df['image_name'].tolist()
        self.embeddings = self.df.drop(columns=['image_name']).values  
        
        # Convert embeddings to a numpy array (if not already)
        self.all_embeddings = np.array(self.embeddings)
        
        # Initialize the HNSWlib index for cosine similarity (can also use 'l2' for Euclidean distance)
        dim = self.all_embeddings.shape[1]  # Dimension of embeddings, e.g., 512
        num_elements = self.all_embeddings.shape[0]  # Number of embeddings

        # Create an HNSW index
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(max_elements=num_elements, ef_construction=100, M=6)

        # Add embeddings to the index
        self.index.add_items(self.all_embeddings, np.arange(num_elements))  # Use index values as labels

        # Set parameters for the query time
        self.index.set_ef(50)  # Higher ef for more accurate results

    def find_similar_images(self, query_embedding, k=5):
        """ Find the top k most similar images to the query embedding """

        # Retrieve the nearest neighbors for a query image
        labels, distances = self.index.knn_query(query_embedding, k=k)
        similar_images = [self.image_names[i] for i in labels[0]]

        return similar_images, distances
