from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np

# Step 1: Load embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-dim embeddings

# Step 2: Corpus (docs)
documents = [
    "Haystack is an open-source framework by deepset for building search and RAG apps.",
    "Retrieval-Augmented Generation (RAG) retrieves relevant passages and uses them to answer questions.",
    "Semantic search uses embeddings to find contextually similar passages."
]

# Step 3: Embed documents
doc_embeddings = embedder.encode(documents, convert_to_tensor=True, normalize_embeddings=True)

# Step 4: Load local LLM (FLAN-T5)
model_name = "google/flan-t5-small"  # Small, CPU-friendly; can use 'flan-t5-base' if your laptop handles it
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Step 5: Ask a query
query = "What is RAG and how does Haystack help build it?"
query_embedding = embedder.encode([query], convert_to_tensor=True, normalize_embeddings=True)

# Step 6: Retrieve top-k docs
similarities = torch.matmul(query_embedding, doc_embeddings.T)
top_k = torch.topk(similarities, k=2)
top_docs = [documents[idx] for idx in top_k.indices[0]]

# Step 7: Generate answer based on retrieved context
context = " ".join(top_docs)
prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
outputs = model.generate(**inputs, max_length=150)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nQuery:", query)
print("Top Docs:", top_docs)
print("Answer:", answer)
