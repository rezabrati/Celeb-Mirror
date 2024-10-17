from utils import setup_facenet, get_facenet_embedding, find_similar_images, show_similar_images ,modify_paths



def main():

    query_image_path = "adele.jpg"
    
    # Example usage with FaceNet
    facenet_model = setup_facenet()

    # Assuming you have a function to get the embedding of a query image
    query_embedding = get_facenet_embedding(facenet_model, query_image_path)

    # Find the 5 most similar images
    similar_images, distances = find_similar_images(query_embedding, k=6)

    celebrity_names = [name.split('/')[-2] for name in similar_images]

    similar_images_local = modify_paths(similar_images)
    
    show_similar_images(similar_images_local, similar_images_local, celebrity_names)

    
if __name__== '__main__':
    main()