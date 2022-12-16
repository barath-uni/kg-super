from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin

# Parse input sentences
sentences = ['This is the first sentence.',
             'This is the 1st sentence.',
             'This is the third sentence.',
             'This is the 3rd sentence.']

# Create a TfidfVectorizer and fit it to the sentences
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(sentences)

# Use KMeans to find cluster centroids
kmeans = KMeans(n_clusters=2)
kmeans.fit(vectors)

# Select a cluster centroid at random
cluster_centroid = kmeans.cluster_centers_[0]

# Select values radially outward until 80% of the sentences are selected
selected_sentences = []
for sentence, vector in zip(sentences, vectors):
    print("VECTOR")
    print(vector.reshape(1,-1))
    print(cluster_centroid.reshape(1,-1))
    # Compute the distance between the sentence vector and the cluster centroid
    distance = pairwise_distances(vector.reshape(1,-1), cluster_centroid.reshape(1,-1))
    print("DISTANCE BEFORE SELECTION")
    print(distance)
    # If the distance is within the desired range, add the sentence to the selected sentences
    if distance <= 0.6:
        selected_sentences.append(sentence)

# Print the selected sentences
print(selected_sentences)