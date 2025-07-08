import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from transformers import pipeline

def load_vector_store(index_path='vector_store/faiss_index_sampled.bin', 
                      metadata_path='vector_store/metadata_sampled.pkl', 
                      chunks_path='vector_store/chunks_sampled.pkl'):
    """Load FAISS index and metadata."""
    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    return index, metadata, chunks

def retrieve_chunks(query, index, metadata, chunks, model, top_k=5):
    """Retrieve top-k relevant chunks."""
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), top_k)
    results = [(chunks[i], metadata[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

def generate_answer(query, retrieved_chunks, generator):
    """Generate answer using retrieved chunks."""
    context = "\n".join([chunk[0] for chunk in retrieved_chunks])
    prompt = f"""You are a financial analyst assistant for CreditTrust. Your task is to answer questions about customer complaints. Use only the provided context to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information. 

Context: {context}

Question: {query}

Answer:"""
    response = generator(prompt, max_new_tokens=150, truncation=True)[0]['generated_text']
    return response.split("Answer:")[-1].strip(), retrieved_chunks

def evaluate_rag():
    """Evaluate RAG pipeline with test questions."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    generator = pipeline('text-generation', model='distilgpt2', device=-1)  # CPU
    index, metadata, chunks = load_vector_store()
    
    questions = [
        "Why are people unhappy with BNPL?",
        "What issues are reported with Credit Cards?",
        "Are there complaints about delays in Money Transfers?",
        "What are common problems with Savings Accounts?",
        "Why do customers complain about Personal Loans?"
    ]
    
    evaluation = []
    for question in questions:
        answer, sources = generate_answer(question, retrieve_chunks(question, index, metadata, chunks, model), generator)
        evaluation.append({
            'Question': question,
            'Generated Answer': answer,
            'Retrieved Sources': [f"Source {i+1}: {s[0]} (Product: {s[1]['product']}, ID: {s[1]['complaint_id']})" for i, s in enumerate(sources[:2])],
            'Quality Score': 3,  # Placeholder; adjust manually
            'Comments': 'Review answer relevance manually'
        })
    
    df_eval = pd.DataFrame(evaluation)
    df_eval.to_csv('notebooks/evaluation_table.csv', index=False)
    print("Evaluation table saved to notebooks/evaluation_table.csv")
    return df_eval

if __name__ == "__main__":
    evaluate_rag()