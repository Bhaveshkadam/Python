from PyPDF2 import PdfReader
import pdfplumber
import numpy as np
import psycopg2
import json
import logging
import asyncio
import requests
from functions.database_connection import get_db_connection
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from the PDF, checking for encryption."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            if reader.is_encrypted:
                reader.decrypt('')
            text = ''.join([page.extract_text() for page in reader.pages])
            if text:
                return text
            else:
                raise ValueError("Extracted text is empty or unreadable.")
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        raise

def generate_embeddings(text: str) -> np.ndarray:
    try:
        return embedding_model.encode(text)
    except Exception as e:
        logging.error(f"An error occurred while generating embeddings: {e}")
        return np.array([])

def store_embeddings_in_db(filename: str, embeddings: np.ndarray):
    try:
        conn, cur = get_db_connection()
        sql = """
        INSERT INTO pdf_embeddings (filename, embeddings)
        VALUES (%s, %s::vector)
        ON CONFLICT (filename) DO UPDATE
        SET embeddings = EXCLUDED.embeddings
        """
        cur.execute(sql, (filename, embeddings.tolist()))
        conn.commit()
    except Exception as e:
        logging.error(f"An error occurred while storing embeddings in DB: {e}")
        conn.rollback()

async def process_file(file_path: str, filename: str):
    try:
        await asyncio.sleep(1)

        text = extract_text_from_pdf(file_path)
        if not text:
            raise ValueError("Extracted text is empty.")

        chunk_size = 1000     
        text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] 

        embeddings_list = []
        for chunk in text_chunks:
            chunk_embeddings = generate_embeddings(chunk)
            if chunk_embeddings.size > 0:  
                embeddings_list.append(chunk_embeddings)
            else:
                logging.warning(f"Failed to generate embeddings for chunk: {chunk[:100]}") 

        if not embeddings_list:
            raise ValueError("No valid embeddings were generated.")
        
        embeddings_array = np.array(embeddings_list)
        embeddings = np.mean(embeddings_array, axis=0)


        store_embeddings_in_db(filename, embeddings)
        
        logging.info(f"Processed file: {filename} and stored embeddings.")
    except Exception as e:
        logging.error(f"An error occurred while processing file {filename}: {e}")

def generate_answer(question, content):
    inputs = tokenizer(question + " " + content, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model.generate(**inputs, max_length=100, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer