from fastapi import FastAPI
from routes.route import router as pdf_router  # Import the router from route.py

app = FastAPI()
app.include_router(pdf_router)