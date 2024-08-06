import requests
import json

def get_response(url, headers, payload):

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response

def get_message(response):
    output = ""
    for line in response.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])

        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content

        if body.get("done", False):
            message["content"] = output
            return message