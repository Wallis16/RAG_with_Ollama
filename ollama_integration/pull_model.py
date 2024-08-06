import requests
import json

# Define the URL and payload
url = 'http://localhost:7869/api/pull'
headers = {
    'Content-Type': 'application/json'
}
payload = {
    'name': 'phi3:mini'
}

# Send the POST request
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Check the response
if response.status_code == 200:
    print("Success:", response.text)
else:
    print("Error:", response.status_code, response.text)
