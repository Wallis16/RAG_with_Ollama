from ollama_integration import send_prompt
from text_embeddings import rag

# url = 'http://localhost:7869/api/chat'

# headers = {
#         'Content-Type': 'application/json'
#     }

# payload = {
#         'model': 'phi3:mini',
#         'messages': [
#             {
#                 'role': 'user',
#                 'content': 'why sky blue'
#             }
#         ]
#     }

# response = send_prompt.get_response(url, headers, payload)
# message = send_prompt.get_message(response)

print(rag.rag_chatbot("what is anarchy?"))