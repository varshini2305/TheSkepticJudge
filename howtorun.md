To start the two agent services.

```
uvicorn restaurant_agent:app --reload --port 8000
uvicorn learning_agent:app   --reload --port 8001
uvicorn agent_evaluator:app   --reload --port 8002
```

To test the restaurant agent - 

```
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
        "user_id": "emily_garcia",
       "query": "I want a really nice spot with great ambience and photogenic space to host my birthday dinner today at 6 PM"
  }'
```

To test the Learning agent - 
```
curl -X POST http://localhost:8001/learn \
  -H "Content-Type: application/json" \
  -d '{
        "query": "explain the main topics related to distributed middleware",
        "user_id": "aiden"
      }'
```

Testing /recommed/visit-feedback
```
curl -X POST http://localhost:8000/recommend/visit-feedback \
  -H "Content-Type: application/json" \
  -d '{
    "user_name": "Alice Smith",
    "restaurant_name": "DIN95051",
    "visited": true,
    "visit_date": "31-05-2025",
    "experience_rating": 4,
    "remarks": "Great experience!"
  }'
```

To test the eval agent
```
curl -X POST http://localhost:8002/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "learning_agent"
  }'

curl -X POST http://localhost:8002/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "restaurant_agent"
  }'
```