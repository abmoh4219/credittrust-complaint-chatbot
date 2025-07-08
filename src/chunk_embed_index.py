import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

def chunk_narratives(df, chunk_size=500, chunk_overlap=50):
    """Chunk complaint narratives."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = []
    metadata = []
    for idx, row in df.iterrows():
        texts = text_splitter.split_text(row['Consumer complaint narrative'])
        for i, text in enumerate(texts):
            chunks.append(text)
            metadata.append({
                'complaint_id': row['Complaint ID'],
                'product': row['Product'],
                'chunk_index': i
            })
    return chunks, metadata

def embed_and_index(chunks, metadata, model_name='all-MiniLM-L6-v2', output_dir='vector_store'):
    """Generate embeddings and create FAISS index in batches."""
    model = SentenceTransformer(model_name)
    batch_size = 16
    embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=True)
        embeddings.append(batch_embeddings)
        if i % 1000 == 0:
            print(f"Processed {i} chunks")
    
    embeddings = np.vstack(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    # Save index and metadata
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, f'{output_dir}/faiss_index_sampled.bin')
    with open(f'{output_dir}/metadata_sampled.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    with open(f'{output_dir}/chunks_sampled.pkl', 'wb') as f:
        pickle.dump(chunks, f)
    
    return embeddings, index

if __name__ == "__main__":
    # Load filtered dataset and sample
    df = pd.read_csv('data/processed/filtered_complaints.csv').sample(n=50000, random_state=42)
    print(f"Sampled dataset shape: {df.shape}")
    
    # Chunk narratives
    chunks, metadata = chunk_narratives(df)
    print(f"Created {len(chunks)} chunks")
    
    # Embed and index
    embeddings, index = embed_and_index(chunks, metadata)
    print(f"Saved FAISS index and metadata to vector_store/")