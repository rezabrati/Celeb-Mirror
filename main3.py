import streamlit as st
from PIL import Image
import time
from utils import setup_facenet, get_facenet_embedding, find_similar_images, modify_paths, process_image
import os


# Set up the title and description of the app
st.title("Celebrity Look-like Finder")
st.write("Upload an image to find the top celebrity look-likes.")

# Step 1: Upload Image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)


if uploaded_file is not None:
    with col1 :
        # Convert uploaded file to PIL image
        query_image = Image.open(uploaded_file)
        
        image_with_mesh = process_image(query_image)

        # Display the uploaded image in Streamlit
        st.image(image_with_mesh, caption="proccessing", use_column_width=True)

        # Display a "Processing" message below the image with a spinner animation
        with st.spinner('Processing...'):

            # Save the uploaded file temporarily
            query_image_path = os.path.join("temp_query.jpg")
            query_image.save(query_image_path)

            # Step 2: Load the FaceNet model
            facenet_model = setup_facenet()

            # Step 3: Get the embedding of the query image
            query_embedding = get_facenet_embedding(facenet_model, query_image_path)

            # Step 4: Find the 5 most similar images from the dataset
            similar_images, distances = find_similar_images(query_embedding, k=3)

            # Step 5: Modify paths for local file system
            similar_images_local = modify_paths(similar_images)


    with col2:
        # Display a progress bar as the similar images are retrieved
        st.write("Retrieving celebrity look-alikes...")

        # Initialize the progress bar
        progress_bar = st.progress(0)

        # Simulate progress over 5 seconds (5 steps, 1 step per second)
        for percent_complete in range(100):
            time.sleep(0.02)  # Slows down the progress bar to complete in 5 seconds
            progress_bar.progress(percent_complete + 1)

        # Step 6: Display the retrieved images as celebrity matches
        st.write("Here are the top 3 celebrity matches:")

        # Display three images side by side with their captions
        cols = st.columns(3)  # Create 3 columns for displaying images
        for i, col in enumerate(cols):
            if i < len(similar_images_local):

                
                # Display each similar image along with the celebrity name
                img = Image.open(similar_images_local[i])  # Open each image
                desired_height = 700
                desired_width = 700
                img = img.resize((desired_width, desired_height))

                celebrity_name = similar_images_local[i].split("/")[6]  # Get celebrity name from path
                col.image(img, caption=f"Look like {i + 1}: {celebrity_name}", use_column_width=True)
            else:
                col.write("No more matches available.")  # If fewer matches, display placeholder text


