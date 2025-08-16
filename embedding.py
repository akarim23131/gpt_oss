from sentence_transformers import SentenceTransformer

with open("input_text_for_embedding.txt", "r") as file:
    content = file.read()
    
lines = content.strip().split('\n')
parts = content.split("Output: ")

recent_input = None
for line in reversed(lines):
    if line.startswith("Input: "):
        recent_input = line.replace("Input: ", "")
        break
    
if recent_input:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(recent_input)
    
    with open("input_query_embeddings.txt", "a") as file:
        file.write(f"{embeddings}\n")
# print(embeddings)


if len(parts) > 1:
    all_output_text = "Output: ".join(parts[1:])  
    
    # Remove any "Input: " lines that might be mixed in
    lines = all_output_text.split('\n')
    filtered_lines = []
    for line in lines:
        if not line.startswith("Input: ") and not line.startswith("--"):
            filtered_lines.append(line)
    
    long_sentence = " ".join(filtered_lines)
    
    
    
    words = long_sentence.split()
    chunk_size = 400
    overlap = 100
    chunks = []
    
    start = 0
    chunk_num = 1
    
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
        with open("output_chunks.txt", "a") as file:
            file.write(f"Sentence {chunk_num}: {chunk_text}\n")
            
        chunks.append(chunk_text)
        chunk_num += 1
        
        start += (chunk_size - overlap)
        
        if end >= len(words):
           break
            
if chunks:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    for i, chunk_text in enumerate(chunks, 1):
        embedding = model.encode(chunk_text)
        
        with open("output_embeddings.txt", "a") as file:
            file.write(f"Sentence {i}: {embedding}\n")
            

    
    







    




