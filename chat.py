import faiss
import pickle
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

print("Loading vector store...")
index = faiss.read_index("vector_store.index")
with open("all_docs.pkl", "rb") as f:
    all_docs = pickle.load(f)

print("Ready!")

def get_query_embedding(query):
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = clip_model.text_model(**{k: v for k, v in inputs.items() if k != 'pixel_values'})
        query_embedding = outputs.pooler_output
    query_np = query_embedding.cpu().numpy().astype("float32")
    if query_np.shape[1] > 512:
        query_np = query_np[:, :512]
    elif query_np.shape[1] < 512:
        query_np = np.pad(query_np, ((0,0), (0, 512 - query_np.shape[1])))
    return query_np

def search(query, top_k=3):
    query_np = get_query_embedding(query)
    scores, indices = index.search(query_np, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        doc = all_docs[idx]
        results.append({"score": scores[0][i], "doc": doc})
    return results

def ask(query):
    results = search(query)
    context = ""
    for r in results:
        doc = r["doc"]
        if doc["type"] == "text":
            context += f"[Page {doc['page']}]: {doc['content'][:500]}\n\n"
        else:
            context += f"[Image on page {doc['page']}]\n\n"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer based on the provided PDF context only."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

print("\n🤖 Multimodal RAG Chatbot Ready!\n")
while True:
    query = input("You: ")
    if query.lower() == "q":
        break
    answer = ask(query)
    print(f"\n🤖 AI: {answer}\n")