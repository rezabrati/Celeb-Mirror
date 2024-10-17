import matplotlib.pyplot as plt
from PIL import Image
import mediapipe as mp
import numpy as np
import os
import cv2

# Define your local base path where the images are stored
BASE_PATH = "C:/Users/vcc/Desktop/celeb-face"

def modify_paths(image_paths):
    """ Modify the paths to point to the local directory """
    local_paths = []
    for path in image_paths:
        # Remove the Google Drive path part and replace with the local base path
        local_path = path.replace("/content/drive/MyDrive/", BASE_PATH + "/")
        local_paths.append(local_path)
    return local_paths

# Function to display top similar images in a 2x3 grid
def show_similar_images(similar_images, image_names, title_names, rows=2, cols=3):
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


# Define a function to apply the face mesh to the uploaded image
def process_image(image):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Define the silver color for drawing the mesh
    silver_color = (220, 220, 192)
    custom_drawing_spec = mp_drawing.DrawingSpec(color=silver_color, thickness=1)
    # Convert the image to RGB
    rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    
    # Process the image with face mesh
    result = face_mesh.process(rgb_image)
    
    # Convert back to BGR for OpenCV
    image_with_mesh = np.array(image)
    
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Draw the face mesh on the image
            mp_drawing.draw_landmarks(
                image=image_with_mesh,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,  # Draw full face mesh
                landmark_drawing_spec=None,
                connection_drawing_spec=custom_drawing_spec  # Silver mesh
            )
    return image_with_mesh