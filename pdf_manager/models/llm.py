import os
from transformers import pipeline

def llms():

    llms = {
        "gpt2": pipeline("text-generation", model=os.getenv("GPT2_MODEL")),
        "bart": pipeline("text2text-generation", model=os.getenv("BART_MODEL"))
    }
    return llms