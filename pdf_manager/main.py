from fastapi import FastAPI, UploadFile, File
import os

app = FastAPI()

UPLOAD_DIRECTORY = "./files/"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

@app.post("/pdf_upload/")
async def upload_pdf(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    
    return {
        "filename": file.filename, "message": "File uploaded successfully on " + UPLOAD_DIRECTORY
        }