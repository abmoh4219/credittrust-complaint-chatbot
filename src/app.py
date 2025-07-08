import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from transformers import pipeline

def load_vector_store(index_path='vector_store/faiss_index_sampled.bin', 
                      metadata_path='vector_store/metadata_sampled.pkl', 
                      chunks_path='vector_store/chunks_sampled.pkl'):
    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    return index, metadata, chunks

def retrieve_chunks(query, index, metadata, chunks, model, top_k=5):
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), top_k)
    results = [(chunks[i], metadata[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

def generate_answer(query, retrieved_chunks, generator):
    context = "\n".join([chunk[0] for chunk in retrieved_chunks])
    prompt = f"""You are a financial analyst assistant for CreditTrust. Your task is to answer questions about customer complaints. Use only the provided context to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information. 

Context: {context}

Question: {query}

Answer:"""
    response = generator(prompt, max_length=200, num_return_sequences=1, truncation=True)[0]['generated_text']
    return response.split("Answer:")[-1].strip(), retrieved_chunks

st.title("CreditTrust Complaint Analysis Chatbot")
st.write("Ask questions about customer complaints (e.g., 'Why are people unhappy with BNPL?')")

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
generator = pipeline('text-generation', model='distilgpt2', device=-1)  # -1 for CPU
index, metadata, chunks = load_vector_store()

query = st.text_input("Enter your question:")
if st.button("Submit"):
    if query:
        answer, sources = generate_answer(query, retrieve_chunks(query, index, metadata, chunks, model), generator)
        st.write("**Answer:**")
        st.write(answer)
        st.write("**Retrieved Sources:**")
        for i, (chunk, meta, _) in enumerate(sources[:2]):
            st.write(f"Source {i+1}: {chunk} (Product: {meta['product']}, Complaint ID: {meta['complaint_id']})")
if st.button("Clear"):
    st.session_state.clear()