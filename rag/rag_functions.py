from ollama_utilities.send_prompt import get_message, get_response
from text_embeddings.search_by_embeddings import search


def format_prompt(prompt,retrieved_documents,k):
  """using the retrieved documents we will prompt the model to generate our responses"""
  PROMPT = f"Question:{prompt}\nContext:"
  for idx in range(k) :
    PROMPT+= f"{retrieved_documents['text'][idx]}\n"
  return PROMPT

def generate(formatted_prompt, SYS_PROMPT, payload, url, headers):
  formatted_prompt = formatted_prompt[:2000]
  messages = f"{SYS_PROMPT} \n \n {formatted_prompt}"
  payload['messages'][0]['content'] = messages
  response = get_response(url, headers, payload)
  message = get_message(response)
  return message

def rag_chatbot(ST, data, SYS_PROMPT, prompt:str,k:int=2):
  _, retrieved_documents = search(ST, data, prompt, k)
  formatted_prompt = format_prompt(prompt,retrieved_documents,k)
  return generate(formatted_prompt, SYS_PROMPT)