from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str
    filename: str = None
    llm: str = "gpt2"  

class QuestionResponse(BaseModel):
    filename: str
    score: float
    content: str
    llm_answer: str