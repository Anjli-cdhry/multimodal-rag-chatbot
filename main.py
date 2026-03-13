import fitz
import torch
import io
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel

print("Multimodal RAG project started")

# ===============================
# Load CLIP Model
# ===============================

print("Loading CLIP model...")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_model.eval()

print("CLIP model loaded successfully")


# ===============================
# TEXT EMBEDDING
# ===============================

def embed_text(text):
    words = text.split()
    chunk_size = 70
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    embeddings = []
    for chunk in chunks:
        inputs = clip_processor(
            text=[chunk],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        with torch.no_grad():
            outputs = clip_model.text_model(**{k: v for k, v in inputs.items() if k != 'pixel_values'})
            text_features = outputs.pooler_output  # direct tensor milega
        
        text_features = text_features / torch.norm(text_features, dim=-1, keepdim=True)
        embeddings.append(text_features.cpu().numpy()[0])
    
    return np.mean(embeddings, axis=0)
# ===============================
# IMAGE EMBEDDING
# ===============================

def embed_image(image):
    image = image.convert("RGB")
    
    inputs = clip_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        image_features = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        image_features = image_features.pooler_output  # direct tensor
    
    # Normalize
    image_features = image_features / torch.norm(image_features, dim=-1, keepdim=True)
    
    return image_features.cpu().numpy()[0]

# ===============================
# PROCESS PDF
# ===============================

print("Opening PDF...")

pdf_path = "multimodal_sample.pdf"
doc = fitz.open(pdf_path)

print("Total pages:", len(doc))

all_docs = []
all_embeddings = []

for page_num in range(len(doc)):

    page = doc[page_num]

    # -----------------
    # TEXT
    # -----------------

    text = page.get_text()

    if text.strip():

        print(f"Processing text from page {page_num}")

        embedding = embed_text(text)

        all_docs.append({
            "type": "text",
            "page": page_num,
            "content": text
        })

        all_embeddings.append(embedding)

    # -----------------
    # IMAGES
    # -----------------

    images = page.get_images(full=True)

    for img_index, img in enumerate(images):

        xref = img[0]

        base_image = doc.extract_image(xref)

        image_bytes = base_image["image"]

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        print(f"Processing image {img_index} from page {page_num}")

        embedding = embed_image(image)

        all_docs.append({
            "type": "image",
            "page": page_num,
            "image_index": img_index
        })

        all_embeddings.append(embedding)


print("PDF processing complete")
print("Total items processed:", len(all_docs))
print("Total embeddings created:", len(all_embeddings))

import faiss

# ===============================
# BUILD FAISS INDEX
# ===============================

print("Building FAISS index...")

embedding_dim = all_embeddings[0].shape[0]  # 512 for CLIP

# Fix inhomogeneous embeddings
fixed_embeddings = []
for i, emb in enumerate(all_embeddings):
    if isinstance(emb, np.ndarray) and emb.ndim == 1:
        fixed_embeddings.append(emb)
    else:
        print(f"Skipping bad embedding at index {i}, shape: {np.array(emb).shape}")

all_embeddings = fixed_embeddings
all_docs = all_docs[:len(fixed_embeddings)]

# Fix: sabko same size karo (512)
target_dim = 512
fixed = []
for emb in all_embeddings:
    emb = emb.flatten()
    if emb.shape[0] > target_dim:
        emb = emb[:target_dim]  # trim karo
    elif emb.shape[0] < target_dim:
        emb = np.pad(emb, (0, target_dim - emb.shape[0]))  # pad karo
    fixed.append(emb)

embeddings_matrix = np.array(fixed).astype("float32")
embedding_dim = target_dim

index = faiss.IndexFlatIP(embedding_dim)  # Inner Product = cosine similarity
index.add(embeddings_matrix)

print(f"FAISS index built! Total vectors: {index.ntotal}")

# ===============================
# SAVE INDEX + DOCS
# ===============================

import pickle

faiss.write_index(index, "vector_store.index")

with open("all_docs.pkl", "wb") as f:
    pickle.dump(all_docs, f)

print("Vector store saved!")