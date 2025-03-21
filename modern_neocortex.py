import requests

url = "http://127.0.0.1:5000/api/solve"
payload = {
    "query": "Explain how neural networks work",
    "use_agents": True,
    "show_work": True
}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())