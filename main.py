from fastapi import FastAPI, UploadFile, File
import os
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# AI EMBEDDING MODEL
# -------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# VECTOR DB (FAISS)
# -------------------------
dimension = 384
index = faiss.IndexFlatL2(dimension)

resume_store = []  # metadata storage

# -------------------------
# HOME
# -------------------------
@app.get("/")
def home():
    return {"message": "PRO AI Resume Screening System Running"}

# -------------------------
# EXTRACT TEXT
# -------------------------
def extract_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# -------------------------
# GET EMBEDDING
# -------------------------
def get_embedding(text):
    return model.encode([text])[0]

# -------------------------
# UPLOAD RESUME (STORE VECTOR)
# -------------------------
@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = extract_text(file_path)

    embedding = get_embedding(text)

    # store in FAISS
    index.add(np.array([embedding]))

    resume_store.append({
        "filename": file.filename,
        "text": text
    })

    return {
        "message": "Resume uploaded & embedded",
        "filename": file.filename
    }

# -------------------------
# RAG SEARCH (PRO AI MATCHING)
# -------------------------
@app.post("/search")
def search(job_description: str):

    query_vector = get_embedding(job_description)

    query_vector = np.array([query_vector])

    distances, indexes = index.search(query_vector, len(resume_store))

    results = []

    for i, idx in enumerate(indexes[0]):
        if idx < len(resume_store):

            results.append({
                "rank": i + 1,
                "filename": resume_store[idx]["filename"],
                "similarity_score": float(distances[0][i])
            })

    return {
        "job_description": job_description,
        "top_candidates": results
    }