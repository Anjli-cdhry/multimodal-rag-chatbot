import streamlit as st
import faiss
import pickle
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from groq import Groq
from dotenv import load_dotenv
from PIL import Image
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="Multimodal RAG Chatbot", page_icon="🤖", layout="wide")

@st.cache_resource
def load_models():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    index = faiss.read_index("vector_store.index")
    with open("all_docs.pkl", "rb") as f:
        all_docs = pickle.load(f)
    return clip_model, clip_processor, index, all_docs

clip_model, clip_processor, index, all_docs = load_models()

def get_text_embedding(query):
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = clip_model.text_model(**{k: v for k, v in inputs.items() if k != 'pixel_values'})
        embedding = outputs.pooler_output
    emb_np = embedding.cpu().numpy().astype("float32")
    if emb_np.shape[1] > 512:
        emb_np = emb_np[:, :512]
    elif emb_np.shape[1] < 512:
        emb_np = np.pad(emb_np, ((0,0), (0, 512 - emb_np.shape[1])))
    return emb_np

def get_image_embedding(image):
    image = image.convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        embedding = image_features.pooler_output
    emb_np = embedding.cpu().numpy().astype("float32")
    if emb_np.shape[1] > 512:
        emb_np = emb_np[:, :512]
    elif emb_np.shape[1] < 512:
        emb_np = np.pad(emb_np, ((0,0), (0, 512 - emb_np.shape[1])))
    return emb_np

def search(embedding, top_k=3):
    scores, indices = index.search(embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({"score": scores[0][i], "doc": all_docs[idx]})
    return results

def ask(query):
    embedding = get_text_embedding(query)
    results = search(embedding)
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

# ========================
# SIDEBAR — Image Search
# ========================
with st.sidebar:
    st.title("🖼️ Image Search")
    st.caption("Upload an image to find similar content in the PDF")
    
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Search Similar Content"):
            with st.spinner("Searching..."):
                emb = get_image_embedding(image)
                results = search(emb, top_k=3)
            
            st.subheader("Top Matches:")
            for i, r in enumerate(results):
                doc = r["doc"]
                st.markdown(f"**Result {i+1}** (Score: {r['score']:.2f})")
                st.markdown(f"📄 Type: `{doc['type']}` | Page: `{doc['page']}`")
                if doc["type"] == "text":
                    st.info(doc["content"][:300])
                else:
                    st.success(f"🖼️ Image found on page {doc['page']}")
                st.divider()

# ========================
# MAIN — Chat UI
# ========================
st.title("🤖 Multimodal RAG Chatbot")
st.caption("Powered by CLIP + FAISS + LLaMA")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I'm your PDF Assistant. I've analyzed the document and I'm ready to help!\n\nYou can ask me things like:\n- 📄 *What is this document about?*\n- 📊 *Summarize the key findings*\n- 🔍 *What does page 9 contain?*\n- 🖼️ *What images are in the document?*"
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(prompt)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})