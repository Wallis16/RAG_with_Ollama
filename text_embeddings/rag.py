from ollama_integration import send_prompt

import os
from dotenv import load_dotenv

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

def search(ST, data, query: str, k: int = 3):
    """a function that embeds a new query and returns the most probable results"""
    embedded_query = ST.encode(query) # embed new query
    scores, retrieved_examples = data.get_nearest_examples( # retrieve results
        "embeddings", embedded_query, # compare our new embedded query with the dataset embeddings
        k=k # get only top k results
    )
    return scores, retrieved_examples

def format_prompt(prompt,retrieved_documents,k):
  """using the retrieved documents we will prompt the model to generate our responses"""
  PROMPT = f"Question:{prompt}\nContext:"
  for idx in range(k) :
    PROMPT+= f"{retrieved_documents['text'][idx]}\n"
  return PROMPT

def generate(formatted_prompt, SYS_PROMPT):
  formatted_prompt = formatted_prompt[:2000]
  messages = f"{SYS_PROMPT} \n \n {formatted_prompt}"
  payload['messages'][0]['content'] = messages
  response = send_prompt.get_response(url, headers, payload)
  message = send_prompt.get_message(response)
  return message

def rag_chatbot(ST, data, prompt:str,k:int=2):
  _, retrieved_documents = search(ST, data, prompt, k)
  formatted_prompt = format_prompt(prompt,retrieved_documents,k)
  return generate(formatted_prompt, SYS_PROMPT)