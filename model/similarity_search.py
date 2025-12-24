import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------- STEP 1: Load Features ----------
with open("database/features.pkl", "rb") as f:
    features_dict = pickle.load(f)

# ---------- STEP 2: Prepare Data ----------
# Convert dictionary to list of filenames and matrix of feature vectors
filenames = list(features_dict.keys())
feature_matrix = np.array([features_dict[f] for f in filenames])

# ---------- STEP 3: Function to Get Top-K Similar Products ----------
def get_top_k_similar(selected_image, k=5):
    if selected_image not in features_dict:
        print("❌ Image not found in database!")
        return []
    
    query_vec = features_dict[selected_image].reshape(1, -1)
    similarities = cosine_similarity(query_vec, feature_matrix)[0]  # similarity with all images

    # Get indices of top-k most similar (exclude itself)
    top_indices = similarities.argsort()[::-1]  # descending order
    top_indices = [i for i in top_indices if filenames[i] != selected_image][:k]

    # Return top-k filenames
    top_images = [filenames[i] for i in top_indices]
    return top_images

# ---------- STEP 4: Test ----------
if __name__ == "__main__":
    test_image = "p2.jpg"  # change this to any image in your folder
    top5 = get_top_k_similar(test_image, k=5)
    print(f"\nTop-5 similar products to '{test_image}':")
    for img in top5:
        print("-", img)
