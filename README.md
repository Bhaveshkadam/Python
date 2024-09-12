## TODO List
 1. Ability to upload large files into a database -DONE
 2. While generating the answer, take a reference to multiple uploaded files. The response must use the first 5 most similar content files. -DONE
 3. Read about Chunking (for uploading large amounts of data) -DONE
 4. Read about background tasks (implement background tasks) -DONE 

 - Chunking embedding refers to the process of breaking down large pieces of text into smaller, more manageable chunks before generating embeddings for each chunk. Embeddings are vector representations of text used in natural language processing (NLP) models to capture the semantic meaning of the text.

 
# Setup Instructions

### To set up and run the project locally, follow these steps:

 1. Prerequisites
Python 3.7+ installed on your system.
   PostgreSQL installed and running.
  Virtual Environment (optional but recommended).

2. Clone the Repository
   Clone the project repository to your local machine:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

3. Set Up a Virtual Environment (Optional)
   Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate 
   ```

4. Install Required Packages
   Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not available, install the dependencies manually:
   ```bash
   pip install fastapi aiofiles psycopg2-binary sentence-transformers scikit-learn pyPDF2 transformers python-multipart  
   ```

 5. Set Up PostgreSQL Database
   - In order to get postgres up and running. Go to the `pdf_management` directory and run `docker-compose up -d`
   - Create a table named `pdf_embeddings` with the following structure:
      ```bash
      CREATE EXTENSION IF NOT EXISTS vector;
      ```
      ```sql
       CREATE TABLE  pdf_embeddings (
      ID SERIAL PRIMARY KEY,
      filename TEXT UNIQUE,
      embeddings VECTOR(384)  -- Adjust the dimensionality based on your embeddings
     );
      ```
6. Run the Application
   Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
   - Access the application at `http://127.0.0.1:8000`.
   - API documentation will be available at `http://127.0.0.1:8000/docs`.

7. Here is the flow of Application APIs

   - Use the `/upload/` endpoint to upload PDF files.
   
      ![alt text](https://github.com/Bhaveshkadam/Python/blob/main/Document/Upload.jpeg)
   
   - Use the `/download/` endpoint to download PDF files.
   
      ![alt text](https://github.com/Bhaveshkadam/Python/blob/main/Document/Download.jpeg)
   
   - Use the `/update/` endpoint to update PDF files.
   
       ![alt text](https://github.com/Bhaveshkadam/Python/blob/main/Document/Update.jpeg)
   
   - Use the `/delete/` endpoint to delete PDF files.
   
       ![alt text](https://github.com/Bhaveshkadam/Python/blob/main/Document/Delete.jpeg)
   
   - Use the `/question/` endpoint to question for relevant information.

       ![alt text](https://github.com/Bhaveshkadam/Python/blob/main/Document/Question.jpeg)


