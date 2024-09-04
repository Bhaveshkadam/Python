from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import aiofiles
import fitz  # PyMuPDF
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

logging.basicConfig(level=logging.INFO)

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
        return "".join(page.get_text() for page in doc)

def generate_embeddings(text: str) -> np.ndarray:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text)

def store_embeddings_in_db(filename: str, embeddings: np.ndarray):
    try:
        sql = """
        INSERT INTO pdf_embeddings (filename, embeddings)
        VALUES (%s, %s::vector)
        ON CONFLICT (filename) DO UPDATE
        SET embeddings = EXCLUDED.embeddings
        """
        cur.execute(sql, (filename, embeddings.tolist()))
        conn.commit()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        conn.rollback()

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    file_path = os.path.join(PDF_DIR, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
        await out_file.write(await file.read())

    text = extract_text_from_pdf(file_path)
    embeddings = generate_embeddings(text)
    store_embeddings_in_db(file.filename, embeddings)

    return {"filename": file.filename, "message": "File uploaded successfully"}

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
        await out_file.write(await file.read())

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

llm = pipeline("text-generation", model="gpt2")

class QuestionRequest(BaseModel):
    question: str
    filename: str = None

class QuestionResponse(BaseModel):
    filename: str
    score: float
    content: str
    llm_answer: str

@app.post("/question", response_model=QuestionResponse)
async def quation_answeer(request: QuestionRequest):
    question = request.question
    filename = request.filename

    if not question:
        raise HTTPException(status_code=400, detail="Not empty, please ask a question.")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([question])[0]
    
    if filename:
        cur.execute("SELECT filename, embeddings FROM pdf_embeddings WHERE filename = %s", (filename,))
    else:
        cur.execute("SELECT filename, embeddings FROM pdf_embeddings")
    
    rows = cur.fetchall()

    best_score = float('-inf')
    best_filename = None
    similarity_threshold = 0.3

    for row in rows:
        filename, stored_embedding_str = row
        
        try:
            stored_embedding = np.array(json.loads(stored_embedding_str), dtype=np.float32)
            score = cosine_similarity([query_embedding], [stored_embedding])[0][0]
            logging.info(f"Filename: {filename}, Similarity Score: {score}")

            if score > best_score:
                best_score = score
                best_filename = filename
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error processing stored_embedding for filename {filename}: {e}")
            continue

    if best_filename is None or best_score < similarity_threshold:
        logging.info(f"No relevant information found for question: {question}")
        raise HTTPException(status_code=404, detail="Sorry, I couldn't find relevant information in the documents.")

    pdf_path = os.path.join(PDF_DIR, best_filename)
    best_content = extract_text_from_pdf(pdf_path)

    llm_answer = llm(question + " " + best_content, max_new_tokens=100)[0]['generated_text']

    return QuestionResponse(filename=best_filename, score=best_score, content=best_content, llm_answer=llm_answer)
