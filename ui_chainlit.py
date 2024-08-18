from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rag.rag_functions import rag_chatbot

import chainlit as cl

ST = SentenceTransformer("all-mpnet-base-v2")

hub_path = "diogenes-wallis/wikipedia-all-countries"
dataset = load_dataset(hub_path, revision = "embedded")
data = dataset["train"]
data = data.add_faiss_index("embeddings")

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

k = 5

@cl.step(type="tool")
async def tool(message):
    output = rag_chatbot(ST, data, SYS_PROMPT, message, k, payload, url, headers)
    return output["content"]

@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """

    final_answer = await cl.Message(content="").send()

    # Call the tool
    final_answer.content = await tool(message.content)

    await final_answer.update()