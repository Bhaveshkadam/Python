from pydantic import BaseModel
from typing import List, Optional

class QuestionRequest(BaseModel):
    question: str
    filename: Optional[str] = None

class FileInfo(BaseModel):
    filename: str
    score: float
    content: str

class QuestionResponse(BaseModel):
    top_files: List[FileInfo] 
    llm_answer: str