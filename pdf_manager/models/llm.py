import os
from transformers import pipeline

llms = {
    "gpt2": pipeline("text-generation", model="gpt2"),
    "bart": pipeline("text2text-generation", model="facebook/bart-large")
}
