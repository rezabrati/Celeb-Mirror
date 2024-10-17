import pandas as pd
import numpy as np
import hnswlib
from sklearn.metrics.pairwise import cosine_similarity

# Function to retrieve embeddings from a CSV file
def find_similar_images(query_embedding , k=5):

    df = pd.read_csv("embeddings_celeb150_portrait.csv")
    image_names = df['image_name'].tolist()
    embeddings = df.drop(columns=['image_name']).values  

    # Convert embeddings to a numpy array (if not already)
    all_embeddings = np.array(embeddings)

    # Initialize the HNSWlib index for cosine similarity (can also use 'l2' for Euclidean distance)
    dim = all_embeddings.shape[1]  # Dimension of embeddings, e.g., 512
    num_elements = all_embeddings.shape[0]  # Number of embeddings

    # Create an HNSW index
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=100, M=6)

    # Add embeddings to the index
    index.add_items(all_embeddings, np.arange(num_elements))  # Use index values as labels

    # Set parameters for the query time
    index.set_ef(50)  # Higher ef for more accurate results


    # Example: Function to retrieve the nearest neighbors for a query image
    labels, distances = index.knn_query(query_embedding, k=k)
    similar_images = [image_names[i] for i in labels[0]]

    return similar_images, distances


