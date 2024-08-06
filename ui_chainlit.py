from text_embeddings import rag
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import chainlit as cl

ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

dataset = load_dataset("not-lain/wikipedia",revision = "embedded")
data = dataset["train"]
data = data.add_faiss_index("embeddings") 

@cl.step(type="tool")
async def tool(message):
    output = rag.rag_chatbot(ST, data, message)
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