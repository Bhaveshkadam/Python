from pydantic import BaseModel
from typing import List, Optional, Dict

class QuestionRequest(BaseModel):
    question: str
    filename: Optional[str] = None

class QuestionResponse(BaseModel):
    top_files: List[Dict[str, str]] 
    llm_answer: str