from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
import aiofiles
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging
from functions.database_connection import get_db_connection
from functions.requestresponce import QuestionRequest, QuestionResponse, FileInfo
from functions.utils import extract_text_from_pdf, generate_embeddings, store_embeddings_in_db, process_file, generate_answer

app = FastAPI()
logging.basicConfig(level=logging.INFO)

PDF_DIR = "./files/"
os.makedirs(PDF_DIR, exist_ok=True)

conn, cur = get_db_connection()

@app.post("/upload/")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    file_path = os.path.join(PDF_DIR, file.filename)
    async with aiofiles.open(file_path, 'wb') as out_file:
            await out_file.write(await file.read())
    
    background_tasks.add_task(process_file, file_path, file.filename)

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

    if not question:
        raise HTTPException(status_code=400, detail="Not empty, please ask a question.")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([question])[0]
    
    if filename:
        cur.execute("SELECT filename, embeddings FROM pdf_embeddings WHERE filename = %s", (filename,))
    else:
        cur.execute("SELECT filename, embeddings FROM pdf_embeddings")
    
    rows = cur.fetchall()

    similarity_threshold = 0.5
    file_scores = []

    for row in rows:
        filename, stored_embedding_str = row
        
        try:
            stored_embedding = np.array(json.loads(stored_embedding_str), dtype=np.float32)
            score = cosine_similarity([query_embedding], [stored_embedding])[0][0]
            score = float(score)
            logging.info(f"Filename: {filename}, Similarity Score: {score}")

            if score >= similarity_threshold:
                file_scores.append((filename, score))

        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Error processing stored_embedding for filename {filename}: {e}")
            continue

    file_scores = sorted(file_scores, key=lambda x: x[1], reverse=True)[:4]

    if not file_scores:
        logging.info(f"No relevant information found for question: {question}")
        raise HTTPException(status_code=404, detail="Sorry, I couldn't find relevant information in the documents.")
    
    top_files = []
    for file in file_scores:
        pdf_path = os.path.join(PDF_DIR, file[0])
        content = extract_text_from_pdf(pdf_path)
        top_files.append(FileInfo(
            filename=file[0],  # String
            score=file[1],     # Float
            content=content    # String
        ))
    best_content = top_files[0].content
    llm_answer = generate_answer(question, best_content)

    return QuestionResponse(
    top_files=top_files,
    llm_answer=llm_answer
)