import os
import json
import yaml
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from openai import OpenAI
import google.generativeai as genai
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from math import sqrt

# —————————— Load Config ——————————
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Initialize Gemini client
genai.configure(api_key=cfg["env"]["GEMINI_API_KEY"])
gemini_client = genai.GenerativeModel('gemini-2.0-flash')

# Initialize SentenceTransformer
sentence_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# Initialize MongoDB client
MONGO_URI = cfg["env"]["MONGODB_ATLAS_URI"]
LEARN_DB = cfg["env"]["mongodb"]["learning_agent"]["db"]
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[LEARN_DB]
users_col = db["user_info"]
materials_col = db["learning_materials"]


# —————————— FastAPI Setup ——————————
app = FastAPI()

class LearnRequest(BaseModel):
    query: str
    user_name: str

class NextRequest(BaseModel):
    doc_name: str
    selected_prompt: str
    user_name: str

# —————————— Helpers ——————————

async def generate_sentence_embedding(text: str, precision: str = "float32") -> list:
    """Generate embedding for text using SentenceTransformer."""
    try:
        embedding = sentence_model.encode(text, precision=precision).tolist()
        return embedding
    except Exception as e:
        print(f"Error generating SentenceTransformer embedding: {str(e)}")
        raise
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

async def find_relevant_context(query: str, top_k: int = 3) -> list[dict]:
    """
    Find the most relevant context using vector similarity search and paragraph-level matching.
    
    Args:
        query (str): The search query
        top_k (int): Number of top results to return
    
    Returns:
        list[dict]: List of relevant contexts with their scores
    """
    try:
        # Generate embedding for the query
        query_vec = sentence_model.encode(query).tolist()
        
        # Vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "study_material_vector",
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": 3,
                    "limit": top_k
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "course": 1,
                    "parsed_text": 1,
                    "score": { "$meta": "searchScore" }
                }
            }
        ]
        
        # Execute the search
        results = list(materials_col.aggregate(pipeline))
        
        if not results:
            return []
            
        # Process each document to find best matching paragraphs
        processed_results = []
        for doc in results:
            # Split into paragraphs
            paragraphs = [p.strip() for p in doc["parsed_text"].split("\n\n") if p.strip()]
            
            if not paragraphs:
                # If no distinct paragraphs, use the whole text
                processed_results.append({
                    "text": doc["parsed_text"],
                    "score": doc["score"],
                    "title": doc.get("title", ""),
                    "course": doc.get("course", "")
                })
                continue
                
            # Find best matching paragraph
            best_para = ""
            best_score = -1.0
            
            for para in paragraphs:
                if len(para) < 20:  # Skip very short paragraphs
                    continue
                para_vec = sentence_model.encode(para).tolist()
                sim = cosine_similarity(query_vec, para_vec)
                if sim > best_score:
                    best_score = sim
                    best_para = para
            
            if best_para:
                processed_results.append({
                    "text": best_para,
                    "score": best_score,
                    "title": doc.get("title", ""),
                    "course": doc.get("course", "")
                })
        
        # Sort by score and return top_k results
        processed_results.sort(key=lambda x: x["score"], reverse=True)
        return processed_results[:top_k]
        
    except Exception as e:
        print(f"Error in vector search: {str(e)}")
        return []

async def find_material_by_query(query: str):
    """
    Search 'learning_materials' using Atlas Search with vector search.
    """
    try:
        # Generate embedding for the query
        query_embedding = await generate_sentence_embedding(query)
        
        # First try vector search
        vector_pipeline = [
            {
                "$vectorSearch": {
                    "index": "study_material_vector",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 3,
                    "limit": 1
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "course": 1,
                    "parsed_text": 1,
                    "score": { "$meta": "searchScore" }
                }
            }
        ]
        
        result = list(materials_col.aggregate(vector_pipeline))
        if result:
            return result[0]
            
        # Fallback to basic text search
        text_query = {
            "$or": [
                {"title": {"$regex": query, "$options": "i"}},
                {"course": {"$regex": query, "$options": "i"}}
            ]
        }
        return materials_col.find_one(text_query)
        
    except Exception as e:
        print(f"Error in material search: {str(e)}")
        # Fallback to basic text search
        text_query = {
            "$or": [
                {"title": {"$regex": query, "$options": "i"}},
                {"course": {"$regex": query, "$options": "i"}}
            ]
        }
        return materials_col.find_one(text_query)

def make_context(text: str, max_len: int = 2000):
    """Truncate the parsed_text for the initial prompt."""
    return text[:max_len]

async def call_gemini_system(system_prompt: str):
    """Call Gemini API with the system prompt asynchronously."""
    try:
        response = await asyncio.to_thread(
            gemini_client.generate_content,
            system_prompt
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {str(e)}")

# —————————— Endpoints ——————————

@app.post("/learn")
async def learn(req: LearnRequest):
    # 1) Find the learning material by name keyword
    material = await find_material_by_query(req.query)
    if not material:
        raise HTTPException(status_code=404, detail="No learning material matching that query")

    # 2) Fetch learner's profile for personalization
    learner = users_col.find_one({"name": req.user_name})
    if not learner:
        raise HTTPException(status_code=404, detail="Learner profile not found")

    # 3) Find relevant context using vector search
    relevant_contexts = await find_relevant_context(req.query)
    context_excerpt = "\n\n".join([ctx["text"] for ctx in relevant_contexts])

    # 4) Build Gemini prompt with learner context + material
    system_prompt = (
        f"You are a tutor, that simplifies any topic using the relevant course materials for context and clearly respond to student's query.\n"
        f"The student wants you to '{req.query}' using the relevant context retrieved below:\n"
        f"{context_excerpt}\n\n"
        f"Generate a response for the student's query, keep it concise in under 2-3 sentences, don't use any formatting punctuations, use a short para with 2-3 sentences for the answer. In addition, generate 3 follow-up questions that the student can use to learn more about that topic.\n"
        f"Return your response in this exact JSON format:\n"
        f"{{\"answer\": \"your detailed answer here\", \"follow_up\": [\"question 1\", \"question 2\", \"question 3\"]}}"
    )

    raw = await call_gemini_system(system_prompt)
    return raw
    # try:
    #     return json.loads(raw)
    # except json.JSONDecodeError:
    #     return raw

    # return {"doc_name": material["title"], "prompts": prompts}


@app.post("/learn/next")
async def learn_next(req: NextRequest):
    # 1) Retrieve same learning material by title
    material = materials_col.find_one({"title": req.doc_name})
    if not material:
        raise HTTPException(status_code=404, detail="Learning material not found")

    # 2) Fetch learner's profile again
    learner = users_col.find_one({"name": req.user_name})
    if not learner:
        raise HTTPException(status_code=404, detail="Learner profile not found")

    parsed_text = material["parsed_text"]
    context_excerpt = make_context(parsed_text)

    # 3) Build Gemini prompt to answer selected prompt + follow-up prompts
    # system_prompt = (
    #     f"You are a tutor for a student with profile:\n"
    #     f"- Name: {learner['name']}\n"
    #     f"- Interests: {learner['interests']}\n"
    #     f"- Personality: {learner['personality']}\n"
    #     f"- Background: {learner['background']}\n\n"
    #     f"The student clicked: \"{req.selected_prompt}\" for the query '{material['title']}'.\n\n"
    #     f"Context excerpt:\n{context_excerpt}\n\n"
    #     "Provide a concise answer (1–2 paragraphs) that addresses the student's selected prompt. "
    #     "Then generate 3 new prompts for further exploration. "
    #     "Return a JSON object: {\"answer\": <string>, \"followup_prompts\": [<strings>]}. "
    #     "Do not include anything else."
    # )
    system_prompt = (
        f"You are a tutor, that simplifies any topic using the relevant course materials for context and clearly respond to student's query.\n"
        f"The student wants you to '{req.query}' using the relevant context retrieved below:\n"
        f"{context_excerpt}\n\n"
        f"Generate a response for the student's query, keep it concise in under 2-3 sentences, don't use any formatting punctuations, use a short para with 2-3 sentences for the answer. In addition, generate 3 follow-up questions that the student can use to learn more about that topic.\n"
        f"Return your response in this exact JSON format:\n"
        f"{{\"answer\": \"your detailed answer here\", \"follow_up\": [\"question 1\", \"question 2\", \"question 3\"]}}"
    )
    raw = await call_gemini_system(system_prompt)
    print(f"response format - {type(raw)=}")
    try:
        resp_json = json.loads(raw)
        answer = resp_json.get("answer", "")
        followups = resp_json.get("followup_prompts", [])
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Malformed LLM response")

    return {"answer": answer, "followup_prompts": followups}