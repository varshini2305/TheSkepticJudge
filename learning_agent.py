import os
import json
import yaml
import asyncio
import uuid
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
import google.generativeai as genai
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from math import sqrt
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from agent_evaluator import AgentEvaluator, ActionType, get_embedding
from langchain.prompts import ChatPromptTemplate
from functools import lru_cache
from typing import Optional, Dict

# —————————— Load Config ——————————
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize evaluator for learning agent
evaluator = AgentEvaluator("learning_agent")

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

# Cache for MongoDB queries
@lru_cache(maxsize=1000)
def get_cached_user(user_id: str) -> Optional[Dict]:
    """Get user profile with caching."""
    return users_col.find_one({"user_id": user_id})

@lru_cache(maxsize=1000)
def get_cached_document(doc_name: str) -> Optional[Dict]:
    """Get document with caching."""
    return materials_col.find_one({"title": doc_name})

# —————————— FastAPI Setup ——————————
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LearnRequest(BaseModel):
    query: str
    user_id: str

class NextRequest(BaseModel):
    doc_name: str
    selected_prompt: str
    user_id: str

class FeedbackRequest(BaseModel):
    user_id: str
    doc_name: str
    helpful: bool
    feedback: str

# —————————— Tools ——————————

@tool
def get_context(query: str, doc_name: str = None) -> dict:
    """Get relevant context for the query using vector similarity search.
    If doc_name is provided, retrieves context for that specific document."""
    try:
        if doc_name:
            # Get context for specific document using cache
            doc = get_cached_document(doc_name)
            if not doc:
                return {"error": f"Document {doc_name} not found"}
            return {"contexts": [{
                "text": doc["parsed_text"],
                "score": 1.0,
                "title": doc.get("title", ""),
                "course": doc.get("course", "")
            }]}
        
        # Generate embedding for the query using cached function
        query_vec = get_embedding(query).tolist()
        
        # Vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "study_material_vector",
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": 1,
                    "limit": 1
                }
            },
            {
                "$addFields": {
                    "searchScore": { "$meta": "searchScore" }
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "title": 1,
                    "course": 1,
                    "parsed_text": 1,
                    "searchScore": 1
                }
            }
        ]
        
        # Execute the search
        results = list(materials_col.aggregate(pipeline))
        
        if not results:
            return {"error": "No relevant content found"}
            
        # Process results
        processed_results = []
        for doc in results:
            processed_results.append({
                "text": doc["parsed_text"],
                "score": doc.get("searchScore", 0.0),
                "title": doc.get("title", ""),
                "course": doc.get("course", "")
            })
        
        return {"contexts": processed_results}
        
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}")
        return {"error": f"Error in vector search: {str(e)}"}

@tool
def get_user_profile(user_id: str) -> dict:
    """Get user's learning profile and preferences."""
    user = get_cached_user(user_id)
    if not user:
        return {"error": "User not found"}
    return user

def update_learning_history(user_id: str, doc_name: str, feedback: str = None) -> bool:
    """Update user's learning history with feedback."""
    try:
        update_data = {
            "last_accessed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feedback": feedback
        }
        
        users_col.update_one(
            {"user_id": user_id},
            {
                "$push": {
                    "learning_history": {
                        "doc_name": doc_name,
                        **update_data
                    }
                }
            }
        )
        return True
    except Exception as e:
        logger.error(f"Error updating history: {str(e)}")
        return False

# —————————— Agent Setup ——————————

# Initialize OpenAI model
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    api_key=cfg["env"]["OPENAI_API_KEY"]
)

# Define the system prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a tutor, that simplifies any topic using the relevant course materials for context and clearly respond to student's query.
    The student wants you to answer their query using the relevant context retrieved.
    
    Generate a response for the student's query, keep it concise in under 2-3 sentences, don't use any formatting punctuations, use a short para with 2-3 sentences for the answer. In addition, generate 3 follow-up questions that the student can use to learn more about that topic.
    
    Return your response in this exact JSON format:
    {{
        "answer": "your detailed answer here",
        "follow_up": ["question 1", "question 2", "question 3"]
    }}"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Initialize tools
tools = [get_context, get_user_profile]
agent = create_tool_calling_agent(llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# —————————— Endpoints ——————————

def truncate_text(text: str, max_length: int = 2000) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

@app.post("/learn")
async def learn(req: LearnRequest):
    """Get personalized learning content based on user query."""
    try:
        # Start tracking request
        evaluator.latency_tracker.start_action(ActionType.MODEL_INFERENCE, {"query": req.query})
        
        # Get user profile
        profile_result = get_user_profile.invoke(req.user_id)
        evaluator.latency_tracker.start_action(ActionType.TOOL_USAGE, {"tool": "get_user_profile", "user_id": req.user_id})
        evaluator.latency_tracker.end_action(
            success="error" not in profile_result,
            output=profile_result
        )
        
        if "error" in profile_result:
            raise HTTPException(status_code=404, detail=profile_result["error"])
        
        # Get context using the tool
        evaluator.latency_tracker.start_action(ActionType.TOOL_USAGE, {"tool": "get_context", "query": req.query})
        context_result = get_context.invoke(req.query)
        evaluator.latency_tracker.end_action(
            success="error" not in context_result,
            output=context_result
        )
        
        if "error" in context_result:
            raise HTTPException(status_code=404, detail=context_result["error"])
        
        # Truncate context texts to limit token usage
        truncated_contexts = []
        for ctx in context_result['contexts']:
            truncated_contexts.append({
                "text": truncate_text(ctx['text']),
                "score": ctx['score'],
                "title": ctx['title'],
                "course": ctx['course']
            })
        
        # Prepare input for the agent with truncated context
        input_data = {
            "input": f"""User query: {req.query}
            User profile:
            - Name: {profile_result['name']}
            - Background: {truncate_text(profile_result.get('background', ''))}
            
            Relevant context:
            {json.dumps([ctx['text'] for ctx in truncated_contexts], indent=2)}
            """
        }
        
        # Get response from agent
        evaluator.latency_tracker.start_action(ActionType.MODEL_INFERENCE, {"input": input_data})
        agent_response = agent_executor.invoke(input_data)
        evaluator.latency_tracker.end_action(success=True, output=agent_response)
        
        # Parse the response to ensure it's in the correct format
        try:
            # First try to get the output from the agent response
            if isinstance(agent_response, dict) and "output" in agent_response:
                response_text = agent_response["output"]
                try:
                    # Try to parse the output as JSON
                    response = json.loads(response_text)
                except json.JSONDecodeError:
                    # If not JSON, use the text as is
                    response = {
                        "answer": response_text,
                        "follow_up": []
                    }
            else:
                # If no output field, use the response directly
                response = agent_response
            
            # Ensure we have the required fields
            final_response = {
                "answer": response.get("answer", str(response) if isinstance(response, str) else ""),
                "follow_up": response.get("follow_up", [])
            }
            
        except Exception as e:
            logger.exception("Error parsing agent response")
            final_response = {
                "answer": str(agent_response),
                "follow_up": []
            }
        
        # Update learning history
        try:
            evaluator.latency_tracker.start_action(
                ActionType.TOOL_USAGE,
                {
                    "tool": "update_learning_history",
                    "user_id": req.user_id,
                    "doc_name": context_result['contexts'][0]['title']
                }
            )
            history_result = update_learning_history(
                req.user_id,
                context_result['contexts'][0]['title']
            )
            evaluator.latency_tracker.end_action(
                success=history_result,
                output={"status": "success" if history_result else "failed"}
            )
        except Exception as e:
            logger.exception("Error updating learning history")
            evaluator.latency_tracker.end_action(
                success=False,
                output={"error": str(e)}
            )
        
        # Log the request-response pair
        session_id = evaluator.log_request(
            query=req.query,
            response=json.dumps(final_response)
        )
        
        return final_response
        
    except Exception as e:
        error_msg = str(e)
        logger.exception("error traceback as follows...")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/learn/next")
async def next_step(req: NextRequest):
    """Handle follow-up prompts using the same context."""
    try:
        # Start tracking request
        evaluator.latency_tracker.start_action(ActionType.MODEL_INFERENCE, {"query": req.selected_prompt})
        
        # Get user profile
        evaluator.latency_tracker.start_action(ActionType.TOOL_USAGE, {"tool": "get_user_profile", "user_id": req.user_id})
        profile_result = get_user_profile.invoke(req.user_id)
        evaluator.latency_tracker.end_action(
            success="error" not in profile_result,
            output=profile_result
        )
        
        if "error" in profile_result:
            raise HTTPException(status_code=404, detail=profile_result["error"])
        
        # Get context for the original document
        evaluator.latency_tracker.start_action(ActionType.TOOL_USAGE, {"tool": "get_context", "doc_name": req.doc_name})
        context_result = get_context.invoke(req.doc_name)
        evaluator.latency_tracker.end_action(
            success="error" not in context_result,
            output=context_result
        )
        
        if "error" in context_result:
            raise HTTPException(status_code=404, detail=context_result["error"])
        
        # Prepare input for the agent
        input_data = {
            "input": f"""Follow-up prompt: {req.selected_prompt}
            Original document: {req.doc_name}
            User profile:
            - Name: {profile_result['name']}
            - Background: {profile_result.get('background', '')}
            
            Relevant context:
            {json.dumps([ctx['text'] for ctx in context_result['contexts']], indent=2)}
            """
        }
        
        # Get response from agent
        evaluator.latency_tracker.start_action(ActionType.MODEL_INFERENCE, {"input": input_data})
        response = agent_executor.invoke(input_data)
        evaluator.latency_tracker.end_action(success=True, output=response)
        
        # Log the request-response pair
        session_id = evaluator.log_request(
            query=req.selected_prompt,
            response=json.dumps(response)
        )
        
        return response
        
    except Exception as e:
        error_msg = str(e)
        logger.exception("error traceback as follows...")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/learn/feedback")
async def provide_feedback(req: FeedbackRequest):
    """Update learning history with user feedback."""
    try:
        evaluator.latency_tracker.start_action(
            ActionType.TOOL_USAGE,
            {
                "tool": "update_learning_history",
                "user_id": req.user_id,
                "doc_name": req.doc_name,
                "feedback": req.feedback
            }
        )
        result = update_learning_history(
            req.user_id,
            req.doc_name,
            req.feedback
        )
        evaluator.latency_tracker.end_action(
            success=result,
            output={"status": "success" if result else "failed"}
        )
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to update feedback")
            
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        error_msg = str(e)
        logger.exception("error traceback as follows...")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/evaluate")
async def evaluate_responses():
    """Evaluate all unevaluated responses for the learning agent."""
    try:
        # Start evaluation process
        evaluator.latency_tracker.start_action(
            ActionType.EVALUATION,
            {"action": "evaluate_unevaluated_requests"}
        )
        
        # Call the evaluation method
        evaluation_summary = evaluator.evaluate_unevaluated_requests()
        
        evaluator.latency_tracker.end_action(
            success=True,
            output=evaluation_summary
        )
        
        return {
            "status": "success",
            "message": "Evaluation completed",
            "summary": evaluation_summary
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.exception("Error during evaluation")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/evaluation/summary")
async def get_evaluation_summary(start_time: Optional[str] = None, end_time: Optional[str] = None):
    """Get a summary of all evaluations within a time period."""
    try:
        # Start getting summary
        evaluator.latency_tracker.start_action(
            ActionType.EVALUATION,
            {
                "action": "get_evaluation_summary",
                "start_time": start_time,
                "end_time": end_time
            }
        )
        
        # Get the summary
        summary = evaluator.get_evaluation_summary(start_time, end_time)
        
        evaluator.latency_tracker.end_action(
            success=True,
            output=summary
        )
        
        return {
            "status": "success",
            "summary": summary
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.exception("Error getting evaluation summary")
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)