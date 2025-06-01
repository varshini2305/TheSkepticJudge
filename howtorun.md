To start the two agent services.

```
uvicorn restaurant_agent:app --reload --port 8000
uvicorn learning_agent:app   --reload --port 8001
```

To test the restaurant agent - 

```
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
        "user_name": "Alice Smith",
       "prompt": "I want a vegan dinner tonight at 6pm"
  }'
```

To test the Learning agent - 
```
curl -X POST http://localhost:8001/learn \
  -H "Content-Type: application/json" \
  -d '{
        "query": "explain the main topics related to distributed middleware",
        "user_name": "Aiden"
      }'
```

