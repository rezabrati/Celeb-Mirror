import streamlit as st
from PIL import Image
import time
import os
from utils import ImageUtils, FaceMeshProcessor, FaceNetModel, ImageFinder

# Set up the title and description of the app
st.title("Celebrity Look-alike Finder")
st.write("Upload an image to find the top celebrity look-alikes.")

# Step 1: Upload Image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Initialize the utility classes outside the file uploader block
image_utils = ImageUtils()
facenet_model = FaceNetModel()  
face_mesh_processor = FaceMeshProcessor()
image_finder = ImageFinder()

if uploaded_file is not None:

    col1 , col2 = st.columns(2)

    with col1:
        # Process the uploaded file
        query_image = Image.open(uploaded_file)

        # Process the image with face mesh
        image_with_mesh = face_mesh_processor.process_image(query_image)

        # Display the uploaded image in Streamlit
        st.image(image_with_mesh, caption="Processing", use_column_width=True)
    with col2:
        # Display a "Processing" message below the image with a spinner animation
        with st.spinner('Processing...'):
            # Save the uploaded file temporarily
            query_image_path = os.path.join("temp_query.jpg")
            query_image.save(query_image_path)

            # Step 2: Get the embedding of the query image
            query_embedding = facenet_model.get_facenet_embedding(query_image_path)

            # Step 3: Find the 3 most similar images from the dataset
            similar_images, distances = image_finder.find_similar_images(query_embedding, k=3)

            # Step 4: Modify paths for local file system
            similar_images_local = image_utils.modify_paths(similar_images)

        # Step 5: Display a progress bar as the similar images are retrieved
        st.write("Retrieving celebrity look-alikes...")
        progress_bar = st.progress(0)

        # Simulate progress over a short period (you can adjust timing)
        for percent_complete in range(100):
            time.sleep(0.02)  # Adjust the speed as needed
            progress_bar.progress(percent_complete + 1)

        # Step 6: Display the retrieved images as celebrity matches
        st.write("Here are the top 3 celebrity matches:")
        cols = st.columns(3)  # Create 3 columns for displaying images

        for i, col in enumerate(cols):
            if i < len(similar_images_local):
                # Display each similar image along with the celebrity name
                img = Image.open(similar_images_local[i])  # Open each image
                desired_height = 700
                desired_width = 700
                img = img.resize((desired_width, desired_height))

                # Assuming the celebrity name is part of the path
                celebrity_name = similar_images_local[i].split("/")[-2]  # Get celebrity name from path
                col.image(img, caption=f"Look like {i + 1}: {celebrity_name}", use_column_width=True)
            else:
                col.write("No more matches available.")  # If fewer matches, display placeholder text
