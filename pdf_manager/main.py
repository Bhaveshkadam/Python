from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import aiofiles
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging
from transformers import pipeline
from functions.requestresponce import QuestionRequest, QuestionResponse
from functions.utils import extract_text_from_pdf, generate_embeddings, store_embeddings_in_db
from models.llm import llms
from functions.database_connection import get_db_connection

app = FastAPI()

logging.basicConfig(level=logging.INFO)

PDF_DIR = "./files/"
os.makedirs(PDF_DIR, exist_ok=True)

conn, cur = get_db_connection()


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

@app.post("/question", response_model=QuestionResponse)
async def quation_answeer(request: QuestionRequest):
    question = request.question
    filename = request.filename
    selected_llm = request.llm

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

    if selected_llm not in llms:
        raise HTTPException(status_code=400, detail="Selected LLM is not available.")

    llm_pipeline = llms[selected_llm]

    # Generate the answer using the selected LLM pipeline
    if selected_llm == "bart":
        llm_answer = llm_pipeline(f"{question} {best_content}", max_length=100, num_beams=4, early_stopping=True)[0]['generated_text']
    else:
        llm_answer = llm_pipeline(question + " " + best_content, max_new_tokens=100)[0]['generated_text']
        
    return QuestionResponse(filename=best_filename, score=best_score, content=best_content, llm_answer=llm_answer)
