import fitz  # PyMuPDF
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
import json
import logging
import asyncio
from functions.database_connection import get_db_connection
conn, cur = get_db_connection()


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

async def process_file(file_path: str, filename: str):
    try:
        await asyncio.sleep(1)

        text = extract_text_from_pdf(file_path)
        embeddings = generate_embeddings(text)
        store_embeddings_in_db(filename, embeddings)
        
        logging.info(f"Processed file: {filename} and stored embeddings.")
    except Exception as e:
        logging.error(f"An error occurred while processing file {filename}: {e}")