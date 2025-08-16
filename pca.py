from sklearn.decomposition import PCA
import numpy as np

# Read input query embeddings (ONE embedding across multiple lines)
with open("input_query_embeddings.txt", "r") as file:
    content = file.read().strip()

# Parse the single input embedding
embedding_str = content.replace('[', '').replace(']', '').replace('\n', ' ')
input_embedding = np.fromstring(embedding_str, sep=' ')
input_embeddings = [input_embedding]

# Read output embeddings (multiple embeddings, each across multiple lines)
with open("output_embeddings.txt", "r") as file:
    content = file.read()

# Split by "Sentence X:" to get each sentence's embedding
sentences = content.split('Sentence ')[1:]  # Skip empty first element
output_embeddings = []

for sentence in sentences:
    # Extract the embedding part (everything after the colon)
    embedding_part = sentence.split(':', 1)[1]
    # Clean and parse
    embedding_str = embedding_part.replace('[', '').replace(']', '').replace('\n', ' ').strip()
    if embedding_str:  # Make sure it's not empty
        embedding = np.fromstring(embedding_str, sep=' ')
        output_embeddings.append(embedding)

# Combine ALL embeddings for PCA
all_embeddings = input_embeddings + output_embeddings
embeddings_array = np.array(all_embeddings)

# Apply PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_array)

# Split results
input_pca = embeddings_2d[:len(input_embeddings)]
output_pca = embeddings_2d[len(input_embeddings):]

# Save results
with open("pca_embeddings.txt", "w") as file:
    file.write("INPUT EMBEDDINGS:\n")
    for i, (x, y) in enumerate(input_pca):
        file.write(f"Input {i+1}: {x}, {y}\n")
    
    file.write("\nOUTPUT EMBEDDINGS:\n")
    for i, (x, y) in enumerate(output_pca):
        file.write(f"Sentence {i+1}: {x}, {y}\n")

print(f"PCA embeddings saved: {len(input_pca)} input + {len(output_pca)} output")