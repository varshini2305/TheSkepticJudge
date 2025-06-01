import os
import yaml
import requests
import uuid
import logging
import json
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, Field
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
from datetime import datetime
from agent_evaluator import AgentEvaluator, ActionType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Initialize MongoDB
MONGO_URI = cfg["env"]["MONGODB_ATLAS_URI"]
RESTAURANT_DB = cfg["env"]["mongodb"]["restaurant_agent"]["db"]
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[RESTAURANT_DB]
users_col = db["user_info"]

# Initialize evaluator for restaurant agent
evaluator = AgentEvaluator("restaurant_agent")

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RestaurantQuery(BaseModel):
    query: str
    user_id: str

class VisitFeedback(BaseModel):
    user_id: str
    restaurant_id: str
    visited: bool
    visit_date: str
    experience_rating: int  # 1-5 scale
    remarks: str

    @validator('visit_date')
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("visit_date must be in YYYY-MM-DD format")

    @validator('experience_rating')
    def validate_rating(cls, v):
        if not 1 <= v <= 5:
            raise ValueError("experience_rating must be between 1 and 5")
        return v

class FeedbackData(BaseModel):
    user_id: str
    restaurant_id: str
    visited: bool
    visit_date: str
    experience_rating: int = Field(ge=1, le=5)
    remarks: str

@tool
def get_user_preferences(user_id: str) -> dict:
    """Get user preferences from the database."""
    user = users_col.find_one({"user_id": user_id})
    if not user:
        return {"error": "User not found"}
    
    # Extract relevant preferences
    preferences = {
        "location": user["location"]["city"],
        "food_type": user["preferences"]["food_type"],
        "allergies": user["preferences"]["allergies"],
        "budget_range": user["preferences"]["budget_range_usd"],
        "radius_km": user["preferences"]["location_radius_km"],
        "general_preference": user["preferences"]["general_preference"],
        "history": user.get("history", [])
    }
    return preferences

@tool
def fusion_ai_api(query: str, chat_id: Optional[str] = None) -> dict:
    """
    Calls Yelp Fusion AI API to get local business information and comparisons from a natural language query.

    Args:
    - query: The user's query for local business information.
    - chat_id: Optional chat ID for maintaining conversation history.

    Returns:
    - dict: JSON response from the Yelp Fusion AI API or an error message.
    """
    url = "https://api.yelp.com/ai/chat/v2"
    headers = {
        "Authorization": f"Bearer {cfg['env']['YELP_API_KEY']}",
        "Content-Type": "application/json"
    }
    
    # Prepare request data
    data = {"query": query}
    if chat_id:
        data["chat_id"] = chat_id

    try:
        logger.info(f"Making Fusion AI API request with query: {query}")
        logger.info(f"Request headers: {json.dumps(headers, default=str)}")
        logger.info(f"Request data: {json.dumps(data, default=str)}")
        
        response = requests.post(url, headers=headers, json=data)
        
        # Log response details
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        
        # Get response as JSON
        try:
            response_json = response.json()
            logger.info(f"Response body: {json.dumps(response_json, default=str)}")
            return response_json
        except json.JSONDecodeError:
            error_msg = f"Invalid JSON response: {response.text}"
            logger.exception("error traceback as follows...")
            return {"error": error_msg}
            
    except requests.RequestException as e:
        error_msg = f"API request failed: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            error_msg += f"\nResponse text: {e.response.text}"
        logger.exception("error traceback as follows...")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.exception("error traceback as follows...")
        return {"error": error_msg}


@tool
def update_visit_feedback(feedback: dict) -> bool:
    """Update user's restaurant visit history with feedback about their experience."""
    try:
        # Prepare update data
        update_data = {
            "restaurant_id": feedback["restaurant_id"],
            "visited": feedback["visited"],
            "visit_date": feedback["visit_date"],
            "experience_rating": feedback["experience_rating"],
            "remarks": feedback["remarks"],
            "feedback_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Update user's history
        users_col.update_one(
            {"user_id": feedback["user_id"]},
            {
                "$push": {
                    "history": update_data
                }
            }
        )
        
        # Log the update
        logger.info(f"Updated visit feedback for user {feedback['user_id']} and restaurant {feedback['restaurant_id']}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating visit feedback: {str(e)}")
        return False

# Initialize OpenAI model
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    api_key=cfg["env"]["OPENAI_API_KEY"]
)

# Define the system prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful restaurant recommendation assistant using Yelp's Fusion AI API.
    Use the user's preferences to refine and enhance their queries for better recommendations.
    Consider their dietary restrictions, allergies, and preferences when formulating queries.
    Avoid recommending restaurants they have recently visited (check their history).
    Consider their general preference for atmosphere and experience.
    
    When using the Fusion AI API:
    1. Start with a basic query based on the user's request
    2. Use the chat_id to maintain conversation context
    3. Refine the query based on the API response and user preferences
    4. Ask follow-up questions to narrow down recommendations
    
    Format your response as a JSON with the following structure:
    {{
        "recommendations": [
            {{
                "name": "restaurant name",
                "reason": "why this restaurant matches the user's preferences",
                "address": "restaurant address",
                "rating": "restaurant rating",
                "price_range": "restaurant price range",
                "cuisine": "restaurant cuisine type",
                "dietary_options": ["vegetarian", "vegan", etc],
                "atmosphere": "brief description of the atmosphere"
            }}
        ],
        "summary": "brief summary of what restaurant is chosen, and how to contact them if needed, and why these restaurants were chosen, considering user's preferences and history"
    }}"""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Initialize tools
tools = [get_user_preferences, fusion_ai_api, update_visit_feedback]
# Initialize the agent
agent = create_tool_calling_agent(llm, tools=tools, prompt=prompt)
# Initialize AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.post("/recommend")
async def recommend_restaurants(req: RestaurantQuery):
    """Get restaurant recommendations based on user query and preferences."""
    try:
        # Start tracking request
        evaluator.latency_tracker.start_action(ActionType.MODEL_INFERENCE, {"query": req.query})
        
        # Get user preferences
        evaluator.latency_tracker.start_action(ActionType.TOOL_USAGE, {"tool": "get_user_preferences", "user_id": req.user_id})
        preferences = get_user_preferences.invoke(req.user_id)
        evaluator.latency_tracker.end_action(
            success="error" not in preferences,
            output=preferences
        )
        
        if "error" in preferences:
            raise HTTPException(status_code=404, detail=preferences["error"])
        
        # Initialize chat_id as None for first request
        chat_id = None
        
        # Prepare initial query with user preferences
        initial_query = f"""Find me {req.query} in {preferences['location']} that offers {', '.join(preferences['food_type'])} options. 
        I have allergies to {', '.join(preferences['allergies'])} and my budget is {preferences['budget_range']}. 
        I prefer {preferences['general_preference']}."""
        
        # Make initial API call
        evaluator.latency_tracker.start_action(ActionType.TOOL_USAGE, {"tool": "fusion_ai_api", "query": initial_query})
        initial_response = fusion_ai_api.invoke({"query": initial_query})
        evaluator.latency_tracker.end_action(
            success="error" not in initial_response,
            output=initial_response
        )
        
        if "error" in initial_response:
            raise HTTPException(status_code=500, detail=initial_response["error"])
            
        # Extract chat_id from response if available
        if "chat_id" in initial_response:
            chat_id = initial_response["chat_id"]
            logger.info(f"Received chat_id: {chat_id}")
        
        # Prepare input for the agent
        input_data = {
            "input": f"""User query: {req.query}
            User preferences:
            - Location: {preferences['location']}
            - Food types: {', '.join(preferences['food_type'])}
            - Allergies: {', '.join(preferences['allergies'])}
            - Budget: {preferences['budget_range']}
            - General preference: {preferences['general_preference']}
            
            Initial API response:
            {json.dumps(initial_response, indent=2)}
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
                        "recommendations": [],
                        "summary": response_text
                    }
            else:
                # If no output field, use the response directly
                response = agent_response
            
            # Ensure we have the required fields
            final_response = {
                "recommendations": response.get("recommendations", []),
                "summary": response.get("summary", str(response) if isinstance(response, str) else "")
            }
            
        except Exception as e:
            logger.exception("Error parsing agent response")
            final_response = {
                "recommendations": [],
                "summary": str(agent_response)
            }
        
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

@app.post("/recommend/visit-feedback")
async def record_visit_feedback(request: Request):
    """Record user feedback about a restaurant visit."""
    try:
        # Parse request body
        feedback_data = await request.json()
        
        # Validate required fields
        required_fields = ["user_id", "restaurant_id", "visited", "visit_date", "experience_rating", "remarks"]
        for field in required_fields:
            if field not in feedback_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Start tracking feedback update
        evaluator.latency_tracker.start_action(
            ActionType.TOOL_USAGE,
            {
                "tool": "update_visit_feedback",
                "user_id": feedback_data["user_id"],
                "restaurant_id": feedback_data["restaurant_id"]
            }
        )
        
        # Update feedback
        result = update_visit_feedback.invoke(feedback_data)
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)  # Using port 8002 to avoid conflicts