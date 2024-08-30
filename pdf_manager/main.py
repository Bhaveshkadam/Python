from fastapi import FastAPI, UploadFile, File, HTTPException
import os

from fastapi.responses import FileResponse

app = FastAPI()

UPLOAD_DIRECTORY = "./files/"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Uplode File
@app.post("/pdf_upload/") 
async def upload_pdf(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    
    return {
        "filename": file.filename, "message": "File uploaded successfully on " + UPLOAD_DIRECTORY
        }

# Download File
@app.get("/pdf/{filename}/download/")
async def download_pdf(filename: str):
    file_location = os.path.join(UPLOAD_DIRECTORY, filename)
    
    if not os.path.exists(file_location):
        raise HTTPException(status_code=404, detail="File not found please try again")
    
    return FileResponse(file_location, media_type="application/pdf", filename=filename)

# Update File
@app.put("/pdf/{filename}/update/")
async def update_pdf(filename: str, file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIRECTORY, filename)
    
    if not os.path.exists(file_location):
        raise HTTPException(status_code=404, detail="File not found")
    
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    
    return {
        "filename": filename, "message": "File updated successfully"
        }

# Delete File
@app.delete("/pdf/{filename}/delete/")
async def delete_pdf(filename: str):
    file_location = os.path.join(UPLOAD_DIRECTORY, filename)
    
    if not os.path.exists(file_location):
        raise HTTPException(status_code=404, detail="File not found")
    
    os.remove(file_location)
    
    return {
        "filename": filename, "message": "File deleted successfully"
        }
