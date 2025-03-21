# NeoCortex API

A simple REST API for interacting with the NeoCortex cognitive system.

## Setup

1. Install the dependencies:
```
pip install -r requirements.txt
```

2. Start the API server:
```
python api.py
```

The server will run on http://localhost:5000 by default.

## Authentication

All API endpoints require authentication with an API key. You can include the API key in one of two ways:

1. As an HTTP header: `X-API-Key: your-api-key`
2. As a query parameter: `?api_key=your-api-key`

The current API key is: `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`

## API Endpoints

### Solve a Problem

```
POST /api/solve
```

Processes a query using the full cognitive system.

**Request Body:**
```json
{
  "query": "Your problem or question here",
  "show_work": true,
  "use_agents": true
}
```

### Fast Response

```
POST /api/fast-response
```

Gets a quicker response without the full reasoning process.

**Request Body:**
```json
{
  "query": "Your problem or question here",
  "show_work": true
}
```

### Get Agent Status

```
GET /api/agents/status
```

Returns the current status of all agents in the system.

## Example Usage

```python
import requests

# API key for authentication
api_key = "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"

# Solve a problem
response = requests.post(
    "http://localhost:5000/api/solve", 
    json={"query": "Explain how quantum computing works"},
    headers={"X-API-Key": api_key}
)
result = response.json()
print(result)

# Get a fast response
response = requests.post(
    "http://localhost:5000/api/fast-response", 
    json={"query": "What is 15 + 27?"},
    headers={"X-API-Key": api_key}
)
result = response.json()
print(result)

# Check agent status
response = requests.get(
    "http://localhost:5000/api/agents/status",
    headers={"X-API-Key": api_key}
)
status = response.json()
print(status)
```

## Web Interface

A simple web interface is included for testing the API. Just open `api_test.html` in your browser and you can:

1. Select different endpoints
2. Enter your API key
3. Input your query
4. Toggle options like `show_work` and `use_agents`
5. Send requests and view the results 