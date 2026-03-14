# 🤖 Multimodal RAG Chatbot

A **Multimodal Retrieval-Augmented Generation (RAG)** system that allows users to ask questions about PDF documents using both **text and images**.

The system extracts **text and images from PDFs**, generates **multimodal embeddings using CLIP**, stores them in a **FAISS vector database**, and retrieves the most relevant content to generate answers using **LLaMA 3.3-70B via the Groq API**.

---

## 🚀 Live Demo

Try the application here:

👉 https://huggingface.co/spaces/Anjaligiriraaj/multimodal-rag-chatbot

---

##  Problem Statement

Traditional document QA systems only work with **text data**. However, many real-world documents such as reports, research papers, and manuals contain **important information in images, charts, and diagrams**.

This project solves that limitation by building a **multimodal retrieval system** that can search across both **text and visual content**.

---

## ⚙️ Tech Stack

| Component | Technology |
|--------|-------------|
| Embedding Model | CLIP (openai/clip-vit-base-patch32) |
| Vector Database | FAISS |
| LLM | LLaMA 3.3-70B (via Groq API) |
| UI | Streamlit |
| PDF Processing | PyMuPDF |
| Deployment | Hugging Face Spaces |

---

## Features

- 📄 Ask questions about any PDF document
- 🖼 Retrieve information from **images inside PDFs**
- ⚡ Fast semantic search using **FAISS**
- 🤖 AI-generated responses powered by **LLaMA**
- 💬 Interactive **chat interface**
- 🚀 Deployed and accessible online

---

## 🏗 System Architecture
PDF Document
↓
Text + Image Extraction (PyMuPDF)
↓
CLIP Embeddings
↓
FAISS Vector Database
↓
Similarity Search
↓
Top Relevant Chunks
↓
LLaMA (Groq API)
↓
Final Answer

---

## 📂 Project Structure
multimodal-rag-chatbot/
│
├── app.py # Streamlit user interface
├── main.py # PDF processing and FAISS index creation
├── chat.py # Terminal-based chatbot
├── requirements.txt # Project dependencies
├── multimodal_sample.pdf # Sample PDF document
├── vector_store.index # FAISS vector index
├── all_docs.pkl # Stored document chunks
└── .env # API keys (not included)

---

## 🔍 How It Works

1. The PDF is processed and **text + images are extracted**.
2. Each chunk is converted into **multimodal embeddings using CLIP**.
3. The embeddings are stored inside a **FAISS vector index**.
4. When a user asks a question, the query is embedded.
5. The most relevant chunks are retrieved using **similarity search**.
6. These chunks are passed to **LLaMA (Groq API)** to generate the final answer.

---

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/Anjli-cdhry/multimodal-rag-chatbot.git
cd multimodal-rag-chatbot

Install dependencies:

pip install -r requirements.txt

Create a .env file:

GROQ_API_KEY=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
---
## Run the Application

Run the Streamlit interface:
streamlit run app.py
Or run the terminal chatbot:
python chat.py
---
📊 Example Queries
Try asking questions like:

Give me the summary of this document
What does the chart on page 3 represent?
Explain the diagram in the PDF
---
👩‍💻 Author

Anjali Choudhary

LinkedIn: www.linkedin.com/in/anjali-choudhary-396904331
GitHub: https://github.com/Anjli-cdhry
---
⭐ If you find this project useful

Consider giving the repository a star ⭐

---

# Why this README is strong

It shows:

- **Problem statement**
- **Architecture**
- **Tech stack**
- **How it works**
- **Deployment**
- **Installation**
- **Example queries**

This is exactly what **recruiters and interviewers expect**.

---

# One small improvement you should add

Add **one architecture diagram** later — it makes the repo look **10x more professional**.

If you want, I can also show you **3 small changes that will make this project look like a senior-level GenAI project on GitHub.**
