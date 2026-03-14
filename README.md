# 🤖 Multimodal RAG Chatbot

A Retrieval-Augmented Generation (RAG) system that supports both **text and image queries** on PDF documents using CLIP, FAISS, and LLaMA.

## 🚀 Live Demo
[Try it on Hugging Face Spaces](https://huggingface.co/spaces/Anjaligiriraj/multimodal-rag-chatbot)

## 🛠️ Tech Stack
- **CLIP** (openai/clip-vit-base-patch32) — Multimodal embeddings for text and images
- **FAISS** — Efficient vector similarity search
- **LLaMA 3.3-70B** (via Groq API) — Fast AI-powered responses
- **Streamlit** — Interactive chat interface
- **PyMuPDF** — PDF text and image extraction

##  Features
- 💬 Ask questions about any PDF document
- 🖼️ Upload images to find similar content in the PDF
- ⚡ Fast responses powered by Groq's LLaMA-3.3-70B
- 🔍 Semantic search using CLIP embeddings

## 📁 Project Structure
```
├── app.py          # Streamlit UI
├── main.py         # PDF processing + FAISS index builder
├── chat.py         # Terminal chatbot
├── requirements.txt
└── .env            # API keys (not included)
```

##  How It Works
1. PDF is processed — text and images are extracted
2. CLIP model generates embeddings for each chunk
3. Embeddings are stored in a FAISS vector index
4. User query is embedded and matched via similarity search
5. Top results are passed to LLaMA for a final answer

##  Setup
```bash
pip install -r requirements.txt
```
Add your API key in `.env`:
```
GROQ_API_KEY=your_groq_api_key
```
Run the app:
```bash
streamlit run app.py
```
