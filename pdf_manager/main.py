from chromadb import Embeddings
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import aiofiles
import fitz  # PyMuPDF
from matplotlib.backend_bases import cursors
from matplotlib.widgets import Cursor
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
from sqlalchemy import update
from textblob import Sentence
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from psycopg2.extensions import register_adapter, AsIs

app = FastAPI()

PDF_DIR = "./files/"
os.makedirs(PDF_DIR, exist_ok=True)

conn = psycopg2.connect(
    dbname="pdf_management",
    user="postgres",
    password="qwerty1201",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from the PDF using PyMuPDF."""
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def generate_embeddings(text: str) -> list:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text)
    return embeddings.tolist()

def store_embeddings_in_db(filename: str, embeddings: list):
    try:
        conn = psycopg2.connect(
            dbname="pdf_management", 
            user="postgres", 
            password="qwerty1201", 
            host="localhost", 
            port="5432"
        )
        cursor = conn.cursor()

        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        embeddings = embeddings.tolist()

        sql = """
        INSERT INTO pdf_embeddings (filename, embeddings)
        VALUES (%s, %s)
        ON CONFLICT (filename) DO UPDATE
        SET embeddings = EXCLUDED.embeddings
        """

        cursor.execute(sql, (filename, embeddings))
        conn.commit()
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    file_path = os.path.join(PDF_DIR, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    text = extract_text_from_pdf(file_path)
    embeddings = generate_embeddings(text)
    print(f"Generated embeddings: {embeddings}")  # Debugging line
    store_embeddings_in_db(file.filename, embeddings)

    return {"filename": file.filename , "message": "File updated successfully"}

@app.get("/download/{filename}")
async def download_pdf(filename: str):
    file_path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type='application/pdf', filename=filename)

@app.put("/update/{filename}")
async def update_pdf(filename: str, file: UploadFile = File(...)):
    file_path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)

    text = extract_text_from_pdf(file_path)
    embeddings = generate_embeddings(text)
    store_embeddings_in_db(filename, embeddings)

    return {"status": "success", "message": f"Embeddings for {filename} updated successfully."}


@app.delete("/delete/{filename}")
async def delete_pdf(filename: str):
    file_path = os.path.join(PDF_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    cur.execute("DELETE FROM pdf_embeddings WHERE filename = %s", (filename,))
    conn.commit()

    return {"message": "File deleted successfully"}


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
