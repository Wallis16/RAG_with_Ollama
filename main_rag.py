import os
from dotenv import load_dotenv
from rag.rag_functions import rag_chatbot
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

ST = SentenceTransformer("all-mpnet-base-v2")

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF")

SYS_PROMPT = """You are an assistant for answering questions.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer."""

url = 'http://localhost:7869/api/chat'

headers = {
        'Content-Type': 'application/json'
    }

payload = {
        'model': 'phi3:mini',
        'messages': [
            {
                'role': 'user',
                'content': ''
            }
        ]
    }

dataset = load_dataset("diogenes-wallis/20220301_simple_100_all_mpnet_base_v2",revision = "embedded")
data = dataset["train"]
data = data.add_faiss_index("embeddings")

prompt = "What is the best European country?"
k = 5

rag_chatbot(ST, data, prompt, k)