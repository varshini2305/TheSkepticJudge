import os
import json
import logging
import datetime
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from pymongo import MongoClient
import yaml
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum
import backoff
import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluateRequest(BaseModel):
    agent_name: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Load metaprompt
METAPROMPT_PATH = "data/judge_metaprompt.txt"
try:
    with open(METAPROMPT_PATH, "r") as f:
        METAPROMPT = f.read()
except FileNotFoundError:
    logger.error(f"Metaprompt file not found at {METAPROMPT_PATH}")
    METAPROMPT = ""

# Initialize MongoDB client
MONGO_URI = cfg["env"]["MONGODB_ATLAS_URI"]
EVAL_DB = "eval_agent_db"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[EVAL_DB]

# Initialize collections based on agent type
def get_collections(agent_name: str):
    """Get the appropriate collections for the agent."""
    if agent_name == "learning_agent":
        return {
            "interactions": db["learning_agent_logs"],
            "evaluations": db["learning_agent_logs"]
        }
    elif agent_name == "restaurant_agent":
        return {
            "interactions": db["restaurant_agent_logs"],
            "evaluations": db["restaurant_agent_logs"]
        }
    else:
        raise ValueError(f"Unknown agent type: {agent_name}")

# Global rate limiter
class GlobalRateLimiter:
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.calls = [t for t in self.calls if now - t < 60]
            
            if len(self.calls) >= self.calls_per_minute:
                # Wait until we can make another call
                sleep_time = 60 - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.calls.append(now)

# Initialize global rate limiter
rate_limiter = GlobalRateLimiter(calls_per_minute=60)

# Initialize single OpenAI model instance for all evaluations
eval_llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=cfg["env"]["OPENAI_API_KEY"]
)

# Initialize sentence transformer for semantic similarity
sentence_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# Cache for embeddings
@lru_cache(maxsize=1000)
def get_embedding(text: str) -> np.ndarray:
    """Get embedding for text with caching."""
    return sentence_model.encode(text)

# Rate limiting configuration
RATE_LIMIT_DELAY = 1.0  # seconds between API calls
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 10.0

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=MAX_RETRIES,
    base=INITIAL_BACKOFF,
    max_value=MAX_BACKOFF
)
def get_llm_evaluation(prompt: str) -> str:
    """Get LLM evaluation with retry logic and rate limiting."""
    rate_limiter.wait_if_needed()
    try:
        return eval_llm.invoke(prompt).content.strip()
    except Exception as e:
        logger.error(f"Error in LLM evaluation: {str(e)}")
        raise

class ActionType(Enum):
    """Types of actions that can be tracked."""
    MODEL_INFERENCE = "model_inference"
    TOOL_USAGE = "tool_usage"
    CONTEXT_RETRIEVAL = "context_retrieval"
    CONTEXT_UPDATE = "context_update"
    RESPONSE_GENERATION = "response_generation"
    EVALUATION = "evaluation"

@dataclass
class ActionMetrics:
    """Metrics for a single action."""
    action_type: ActionType
    start_time: float
    end_time: float
    success: bool
    error_message: Optional[str] = None
    input_params: Optional[Dict] = None
    output: Optional[Any] = None
    
    @property
    def latency(self) -> float:
        """Calculate latency in seconds."""
        return self.end_time - self.start_time

class LatencyTracker:
    """Tracks latency and success status for various actions."""
    
    def __init__(self):
        self.actions: List[ActionMetrics] = []
        self.current_action: Optional[ActionMetrics] = None
    
    def start_action(self, action_type: ActionType, input_params: Optional[Dict] = None) -> None:
        """Start tracking a new action."""
        self.current_action = ActionMetrics(
            action_type=action_type,
            start_time=time.time(),
            end_time=0.0,
            success=False,
            input_params=input_params
        )
    
    def end_action(self, success: bool, output: Any = None, error_message: Optional[str] = None) -> ActionMetrics:
        """End tracking the current action."""
        if not self.current_action:
            raise ValueError("No action is currently being tracked")
        
        self.current_action.end_time = time.time()
        self.current_action.success = success
        self.current_action.output = output
        self.current_action.error_message = error_message
        
        self.actions.append(self.current_action)
        action = self.current_action
        self.current_action = None
        return action
    
    def get_action_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked actions."""
        summary = {
            "total_actions": len(self.actions),
            "successful_actions": sum(1 for a in self.actions if a.success),
            "failed_actions": sum(1 for a in self.actions if not a.success),
            "total_latency": sum(a.latency for a in self.actions),
            "action_breakdown": {}
        }
        
        # Group actions by type
        for action_type in ActionType:
            type_actions = [a for a in self.actions if a.action_type == action_type]
            if type_actions:
                summary["action_breakdown"][action_type.value] = {
                    "count": len(type_actions),
                    "success_rate": sum(1 for a in type_actions if a.success) / len(type_actions),
                    "avg_latency": sum(a.latency for a in type_actions) / len(type_actions),
                    "min_latency": min(a.latency for a in type_actions),
                    "max_latency": max(a.latency for a in type_actions)
                }
        
        return summary

class PromptGenerator:
    """Generates high-quality prompts using the metaprompt system."""
    
    def __init__(self):
        self.metaprompt = METAPROMPT
        # Remove duplicate LLM instance
        self.last_api_call = 0.0
    
    def extract_between_tags(self, tag: str, string: str, strip: bool = False) -> List[str]:
        """Extract content between XML-style tags."""
        ext_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
        if strip:
            ext_list = [e.strip() for e in ext_list]
        return ext_list
    
    def remove_empty_tags(self, text: str) -> str:
        """Remove empty XML-style tags from the end of text."""
        return re.sub(r'<(\w+)></\1>$', '', text)
    
    def extract_prompt(self, metaprompt_response: str) -> str:
        """Extract the generated prompt from the metaprompt response."""
        between_tags = self.extract_between_tags("Instructions", metaprompt_response)[0]
        return self.remove_empty_tags(self.remove_empty_tags(between_tags).strip()).strip()
    
    def extract_variables(self, prompt: str) -> Set[str]:
        """Extract variables from the prompt template."""
        pattern = r'{([^}]+)}'
        variables = re.findall(pattern, prompt)
        return set(variables)
    
    def generate_evaluation_prompt(self, metric_name: str, task_description: str) -> Tuple[str, Set[str]]:
        """Generate a high-quality prompt for a specific evaluation metric."""
        task = f"Evaluate the {metric_name} of an AI assistant's response to a user query"
        
        # Prepare the metaprompt with the task
        prompt = self.metaprompt.replace("{{TASK}}", task)
        
        # Add task description and metric-specific context
        prompt += f"\n\nTask Description: {task_description}\n"
        prompt += f"Evaluation Focus: {metric_name}\n"
        
        try:
            response = get_llm_evaluation(prompt)  # Use global rate-limited function
            extracted_prompt = self.extract_prompt(response)
            variables = self.extract_variables(extracted_prompt)
            return extracted_prompt, variables
        except Exception as e:
            logger.error(f"Error generating prompt for {metric_name}: {str(e)}")
            return self._get_default_prompt(metric_name), set()

    def _get_default_prompt(self, metric_name: str) -> str:
        """Get default prompt template for a metric if generation fails."""
        default_prompts = {
            "completeness": """You are an evaluator that scores how complete a response is in addressing a query.
            Score from 0 to 1, where:
            0 = Missing critical information
            0.5 = Partially complete
            1 = Fully complete
            
            Consider:
            1. Does it address all aspects of the query?
            2. Are there any gaps in the information?
            3. Is the depth of information sufficient?
            
            Return only the score as a float.""",
            "coherence": """You are an evaluator that scores how coherent and well-structured a response is.
            Score from 0 to 1, where:
            0 = Disjointed and hard to follow
            0.5 = Somewhat coherent
            1 = Very coherent and well-structured
            
            Consider:
            1. Is the information logically organized?
            2. Are the ideas connected smoothly?
            3. Is it easy to follow the flow of information?
            
            Return only the score as a float.""",
            # Add other default prompts...
        }
        return default_prompts.get(metric_name, "Score the response from 0 to 1.")

class MetaPromptEvaluator:
    """Evaluates and refines prompts for agent tasks."""
    
    def __init__(self):
        self.meta_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a prompt engineering expert. Your task is to refine a brief task description into a detailed, high-quality prompt that will guide an AI agent.
            
            Consider these aspects when refining the prompt:
            1. Task Clarity: Make the objective crystal clear
            2. Context Requirements: Specify what information is needed
            3. Output Format: Define the expected structure and format
            4. Constraints: List any limitations or requirements
            5. Success Criteria: Define what makes a good response
            
            Return a JSON with:
            {
                "refined_prompt": "detailed prompt",
                "key_components": ["list of critical elements"],
                "evaluation_criteria": ["list of specific metrics"]
            }"""),
            ("human", "{task_description}")
        ])
    
    def refine_prompt(self, task_description: str) -> Dict:
        """Refine a brief task description into a detailed prompt."""
        try:
            result = eval_llm.invoke(self.meta_prompt.format(task_description=task_description))
            return json.loads(result.content)
        except Exception as e:
            logger.error(f"Error refining prompt: {str(e)}")
            return {
                "refined_prompt": task_description,
                "key_components": [],
                "evaluation_criteria": []
            }

class HumanFeedbackCollector:
    """Collects and manages human feedback for agent responses."""
    
    def __init__(self, db):
        self.db = db
        self.feedback_collection = db["human_feedback"]
    
    def collect_feedback(self, session_id: str, response_id: str, feedback: Dict[str, Any]) -> bool:
        """Collect feedback for a specific response."""
        try:
            feedback_entry = {
                "session_id": session_id,
                "response_id": response_id,
                "feedback": feedback,
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.feedback_collection.insert_one(feedback_entry)
            return True
        except Exception as e:
            logger.error(f"Error collecting feedback: {str(e)}")
            return False
    
    def get_feedback(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all feedback for a session."""
        try:
            return list(self.feedback_collection.find({"session_id": session_id}))
        except Exception as e:
            logger.error(f"Error getting feedback: {str(e)}")
            return []
    
    def get_feedback_summary(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict[str, Any]:
        """Get a summary of feedback within a time period."""
        query = {}
        if start_time and end_time:
            query["timestamp"] = {
                "$gte": start_time,
                "$lte": end_time
            }
        
        feedback_entries = list(self.feedback_collection.find(query))
        
        summary = {
            "total_feedback": len(feedback_entries),
            "feedback_by_type": {},
            "average_ratings": {}
        }
        
        for entry in feedback_entries:
            feedback = entry["feedback"]
            
            # Count feedback types
            feedback_type = feedback.get("type", "unknown")
            if feedback_type not in summary["feedback_by_type"]:
                summary["feedback_by_type"][feedback_type] = 0
            summary["feedback_by_type"][feedback_type] += 1
            
            # Aggregate ratings
            if "rating" in feedback:
                rating = feedback["rating"]
                if rating not in summary["average_ratings"]:
                    summary["average_ratings"][rating] = 0
                summary["average_ratings"][rating] += 1
        
        return summary

def load_evaluation_prompts(agent_name: str) -> Dict[str, Dict[str, str]]:
    """Load evaluation prompts for a specific agent from JSON file."""
    prompts_file = f"data/evaluation_prompts/{agent_name}_prompts.json"
    try:
        with open(prompts_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Evaluation prompts file not found for {agent_name}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing evaluation prompts for {agent_name}: {str(e)}")
        return {}

class ResponseEvaluator:
    def __init__(self, agent_name: str):
        """Initialize the response evaluator with task-agnostic metrics."""
        self.agent_name = agent_name
        self.metrics = {
            "clarity": self._evaluate_clarity,
            "completeness": self._evaluate_completeness,
            "relevance": self._evaluate_relevance,
            "engagement": self._evaluate_engagement,
            "instruction_following_quality": self._evaluate_instruction_following_quality
        }
        self.evaluation_prompts = load_evaluation_prompts(agent_name)
        self.previous_responses = {}
    
    def _get_evaluation_prompt(self, metric_name: str, response: str, query: str) -> str:
        """Get the evaluation prompt for a metric, filling in variables."""
        if metric_name not in self.evaluation_prompts:
            return self._get_default_prompt(metric_name)
        
        prompt_data = self.evaluation_prompts[metric_name]
        prompt = prompt_data["prompt"]
        
        # Fill in variables
        prompt = prompt.replace("{response}", response)
        prompt = prompt.replace("{query}", query)
        
        return prompt
    
    def _get_default_prompt(self, metric_name: str) -> str:
        """Get default prompt template for a metric if loading fails."""
        default_prompts = {
            "clarity": """You are an evaluator that scores how clear and understandable a response is.
            Score from 0 to 1, where:
            0 = Confusing, uses complex language, or lacks clear explanations
            0.5 = Somewhat clear but could be simplified
            1 = Crystal clear with appropriate language and explanations
            
            Consider:
            1. Is the language simple and accessible?
            2. Are complex concepts or terms explained clearly?
            3. Is the information presented in a structured way?
            
            Return only the score as a float.""",
            "completeness": """You are an evaluator that scores how complete a response is in addressing a query.
            Score from 0 to 1, where:
            0 = Missing critical information
            0.5 = Partially complete
            1 = Fully complete
            
            Consider:
            1. Does it address all aspects of the query?
            2. Are there any gaps in the information?
            3. Is the depth of information sufficient?
            
            Return only the score as a float.""",
            "relevance": """You are an evaluator that scores how relevant and focused a response is.
            Score from 0 to 1, where:
            0 = Off-topic or inappropriate level of detail
            0.5 = Somewhat relevant but could be more focused
            1 = Perfectly targeted to the user's needs and query
            
            Consider:
            1. Does it directly address the specific query?
            2. Is the level of detail appropriate?
            3. Does it stay focused on key requirements?
            
            Return only the score as a float.""",
            "engagement": """You are an evaluator that scores how engaging and user-friendly a response is.
            Score from 0 to 1, where:
            0 = Passive, dry response with no engagement
            0.5 = Some engaging elements but could be more interactive
            1 = Highly engaging with user-friendly elements
            
            Consider:
            1. Does it maintain user interest?
            2. Does it use engaging examples?
            3. Does it encourage interaction?
            
            Return only the score as a float.""",
            "instruction_following_quality": """You are an evaluator that scores how well a response follows instructions.
            Score from 0 to 1, where:
            0 = Violates constraints or guidelines
            0.5 = Partially follows requirements
            1 = Perfectly follows all requirements and constraints
            
            Consider:
            1. Does it follow all guidelines?
            2. Does it respect user preferences?
            3. Does it handle special requirements appropriately?
            
            Return only the score as a float."""
        }
        return default_prompts.get(metric_name, "Score the response from 0 to 1.")
    
    def _evaluate_with_retry(self, metric_name: str, response: str, query: str) -> float:
        """Evaluate a metric with retry logic and rate limiting."""
        prompt = self._get_evaluation_prompt(metric_name, response, query)
        try:
            result = get_llm_evaluation(prompt)
            return float(result)
        except Exception as e:
            logger.error(f"Error in {metric_name} evaluation after retries: {str(e)}")
            return 0.0
    
    def _evaluate_clarity(self, response: str, query: str) -> float:
        """Evaluate if the response is clear and understandable."""
        return self._evaluate_with_retry("clarity", response, query)
    
    def _evaluate_completeness(self, response: str, query: str) -> float:
        """Evaluate if the response covers all necessary aspects of the query."""
        return self._evaluate_with_retry("completeness", response, query)
    
    def _evaluate_relevance(self, response: str, query: str) -> float:
        """Evaluate if the response is relevant and focused on the query."""
        return self._evaluate_with_retry("relevance", response, query)
    
    def _evaluate_engagement(self, response: str, query: str) -> float:
        """Evaluate if the response is engaging and user-friendly."""
        return self._evaluate_with_retry("engagement", response, query)
    
    def _evaluate_instruction_following_quality(self, response: str, query: str) -> float:
        """Evaluate if the response follows the instructions given in the query."""
        return self._evaluate_with_retry("instruction_following_quality", response, query)
    
    def _get_score_level(self, score: float) -> str:
        """Get the level of a score (high/medium/low)."""
        if score >= 0.8:
            return "high"
        elif score >= 0.5:
            return "medium"
        return "low"
    
    def evaluate_response(self, response: str, query: str, start_time: float, end_time: float, 
                         previous_responses: List[str] = None, session_id: str = None,
                         tool_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate a response using all metrics with rate limiting."""
        scores = {}
        evaluation_start = time.time()
        
        # Get previous responses for consistency check
        if previous_responses is None:
            previous_responses = []
        
        # Evaluate quality metrics using LLM
        for metric_name, metric_func in self.metrics.items():
            try:
                scores[metric_name] = metric_func(response, query)
            except Exception as e:
                logger.error(f"Error in {metric_name} evaluation: {str(e)}")
                scores[metric_name] = 0.0
        
        # Calculate overall quality score (weighted average)
        weights = {
            "clarity": 0.25,
            "completeness": 0.25,
            "relevance": 0.20,
            "engagement": 0.15,
            "instruction_following_quality": 0.15
        }
        
        quality_score = sum(scores[metric] * weights[metric] for metric in weights)
        scores["quality"] = quality_score
        
        # Add score levels
        scores["levels"] = {
            metric: self._get_score_level(score)
            for metric, score in scores.items()
        }
        
        # Store response for future consistency checks
        if session_id:
            if session_id not in self.previous_responses:
                self.previous_responses[session_id] = []
            self.previous_responses[session_id].append(response)
        
        # Calculate evaluation latency
        evaluation_latency = time.time() - evaluation_start
        
        # Add performance metrics (non-LLM based)
        performance_metrics = {
            "total_latency": end_time - start_time,
            "evaluation_latency": evaluation_latency,
            "tool_metrics": tool_metrics or {}
        }
        
        return {
            "scores": scores,
            "performance": performance_metrics,
            "needs_human_feedback": scores["quality"] >= 0.7  # High score threshold
        }
    
    def get_evaluation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of evaluations for a session."""
        if session_id not in self.previous_responses:
            return {
                "total_responses": 0,
                "average_scores": {},
                "score_trends": {}
            }
        
        responses = self.previous_responses[session_id]
        return {
            "total_responses": len(responses),
            "score_trends": {
                metric: [self._get_score_level(score) for score in scores]
                for metric, scores in self._calculate_score_trends(session_id).items()
            }
        }
    
    def _calculate_score_trends(self, session_id: str) -> Dict[str, List[float]]:
        """Calculate score trends for each metric."""
        # This would be implemented to track how scores change over time
        # For now, return empty trends
        return {metric: [] for metric in self.metrics.keys()}
    
    def clear_session_data(self, session_id: str) -> None:
        """Clear stored data for a session."""
        if session_id in self.previous_responses:
            del self.previous_responses[session_id]

def log_evaluation(session_id: str, response: str, query: str, scores: Dict[str, float], agent_name: str) -> None:
    """Log evaluation results to the database."""
    collections = get_collections(agent_name)
    evaluation = {
        "session_id": session_id,
        "response": response,
        "query": query,
        "scores": scores,
        "timestamp": datetime.datetime.now().isoformat(),
        "type": "evaluation"
    }
    collections["evaluations"].insert_one(evaluation)

def get_agent_performance(agent_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
    """Get performance metrics for an agent over a time period."""
    collections = get_collections(agent_name)
    query = {"agent_name": agent_name}
    if start_date and end_date:
        query["start_time"] = {
            "$gte": start_date,
            "$lte": end_date
        }
    
    sessions = list(collections["interactions"].find(query))
    evaluations = list(collections["evaluations"].find({
        "session_id": {"$in": [s["session_id"] for s in sessions]},
        "type": "evaluation"
    }))
    
    # Calculate aggregate metrics
    metrics = {
        "total_sessions": len(sessions),
        "total_responses": len(evaluations),
        "average_scores": {},
        "tool_usage": {},
        "latency_stats": {
            "average": 0.0,
            "min": float('inf'),
            "max": 0.0
        }
    }
    
    if evaluations:
        # Calculate average scores
        score_fields = ["completeness", "coherence", "accuracy", "consistency", 
                       "instruction_obeying", "latency", "overall"]
        for field in score_fields:
            scores = [e["scores"].get(field, 0) for e in evaluations]
            metrics["average_scores"][field] = sum(scores) / len(scores)
        
        # Calculate tool usage statistics
        tool_counts = {}
        for session in sessions:
            for tool in session.get("tool_usages", []):
                tool_name = tool["tool_name"]
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        metrics["tool_usage"] = tool_counts
        
        # Calculate latency statistics
        latencies = [e["scores"].get("latency", 0) for e in evaluations]
        if latencies:
            metrics["latency_stats"] = {
                "average": sum(latencies) / len(latencies),
                "min": min(latencies),
                "max": max(latencies)
            }
    
    return metrics

class AgentEvaluator:
    def __init__(self, agent_name: str):
        """Initialize the evaluator for a specific agent."""
        self.agent_name = agent_name
        self.collections = get_collections(agent_name)
        self.latency_tracker = LatencyTracker()
        self.feedback_collector = HumanFeedbackCollector(db)
        self.response_evaluator = ResponseEvaluator(agent_name)
    
    def log_request(self, query: str, response: str, session_id: Optional[str] = None) -> str:
        """Log a new request-response pair to the database."""
        timestamp = datetime.datetime.now().isoformat()
        
        # Generate session_id if not provided
        if not session_id:
            session_id = str(datetime.datetime.now().timestamp())
        
        # Get tool usage and latency metrics for this request
        tool_metrics = self._get_current_tool_metrics()
        latency_metrics = self.latency_tracker.get_action_summary()
        
        # Create the log entry
        log_entry = {
            "session_id": session_id,
            "agent_name": self.agent_name,
            "query": query,
            "response": response,
            "timestamp": timestamp,
            "is_evaluated": False,
            "tool_usage": tool_metrics,
            "latency_metrics": latency_metrics,
            "scores": {
                "clarity": None,
                "completeness": None,
                "relevance": None,
                "engagement": None,
                "instruction_following_quality": None,
                "quality": None
            }
        }
        
        # Insert into database
        self.collections["interactions"].insert_one(log_entry)
        
        # Reset latency tracker for next request
        self.latency_tracker = LatencyTracker()
        
        return session_id
    
    def _get_current_tool_metrics(self) -> Dict[str, Any]:
        """Get metrics for tools used in the current request."""
        metrics = {
            "total_tools_used": 0,
            "successful_tools": 0,
            "failed_tools": 0,
            "average_latency": 0.0,
            "tool_breakdown": {}
        }
        
        for action in self.latency_tracker.actions:
            if action.action_type == ActionType.TOOL_USAGE:
                metrics["total_tools_used"] += 1
                if action.success:
                    metrics["successful_tools"] += 1
                else:
                    metrics["failed_tools"] += 1
                
                tool_name = action.input_params.get("tool_name", "unknown")
                if tool_name not in metrics["tool_breakdown"]:
                    metrics["tool_breakdown"][tool_name] = {
                        "count": 0,
                        "success_count": 0,
                        "total_latency": 0.0
                    }
                
                metrics["tool_breakdown"][tool_name]["count"] += 1
                if action.success:
                    metrics["tool_breakdown"][tool_name]["success_count"] += 1
                metrics["tool_breakdown"][tool_name]["total_latency"] += action.latency
        
        if metrics["total_tools_used"] > 0:
            metrics["average_latency"] = sum(
                action.latency for action in self.latency_tracker.actions 
                if action.action_type == ActionType.TOOL_USAGE
            ) / metrics["total_tools_used"]
        
        return metrics
    
    def evaluate_unevaluated_requests(self) -> Dict[str, Any]:
        """Evaluate all requests that haven't been evaluated yet."""
        # Find all unevaluated requests
        unevaluated_requests = list(self.collections["interactions"].find({
            "agent_name": self.agent_name,
            "is_evaluated": False
        }))
        
        evaluation_summary = {
            "total_requests": len(unevaluated_requests),
            "evaluated_requests": 0,
            "average_scores": {
                "clarity": 0.0,
                "completeness": 0.0,
                "relevance": 0.0,
                "engagement": 0.0,
                "instruction_following_quality": 0.0,
                "quality": 0.0
            }
        }
        
        total_scores = {metric: 0.0 for metric in evaluation_summary["average_scores"]}
        
        for request in unevaluated_requests:
            try:
                # Evaluate the response using custom prompts
                evaluation = self.response_evaluator.evaluate_response(
                    response=request["response"],
                    query=request["query"],
                    start_time=datetime.datetime.fromisoformat(request["timestamp"]).timestamp(),
                    end_time=datetime.datetime.now().timestamp(),
                    session_id=request["session_id"],
                    tool_metrics=request["tool_usage"]
                )
                
                # Update the request with evaluation scores
                self.collections["interactions"].update_one(
                    {"_id": request["_id"]},
                    {
                        "$set": {
                            "scores": evaluation["scores"],
                            "is_evaluated": True,
                            "evaluation_timestamp": datetime.datetime.now().isoformat()
                        }
                    }
                )
                
                # Update summary statistics
                evaluation_summary["evaluated_requests"] += 1
                for metric, score in evaluation["scores"].items():
                    if metric in total_scores:
                        total_scores[metric] += score
                
            except Exception as e:
                logger.error(f"Error evaluating request {request['_id']}: {str(e)}")
                continue
        
        # Calculate average scores
        if evaluation_summary["evaluated_requests"] > 0:
            for metric, total in total_scores.items():
                evaluation_summary["average_scores"][metric] = (
                    total / evaluation_summary["evaluated_requests"]
                )
        
        return evaluation_summary
    
    def get_evaluation_summary(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict[str, Any]:
        """Get a summary of all evaluations within a time period."""
        query = {"agent_name": self.agent_name, "is_evaluated": True}
        if start_time and end_time:
            query["timestamp"] = {
                "$gte": start_time,
                "$lte": end_time
            }
        
        evaluated_requests = list(self.collections["interactions"].find(query))
        
        summary = {
            "total_requests": len(evaluated_requests),
            "average_scores": {
                "clarity": 0.0,
                "completeness": 0.0,
                "relevance": 0.0,
                "engagement": 0.0,
                "instruction_following_quality": 0.0,
                "quality": 0.0
            },
            "tool_usage_stats": {},
            "latency_stats": {
                "average": 0.0,
                "min": float('inf'),
                "max": 0.0
            }
        }
        
        if not evaluated_requests:
            return summary
        
        # Calculate average scores
        total_scores = {metric: 0.0 for metric in summary["average_scores"]}
        latencies = []
        
        for request in evaluated_requests:
            # Aggregate scores
            for metric, score in request["scores"].items():
                if metric in total_scores:
                    total_scores[metric] += score
            
            # Aggregate tool usage stats
            for tool_name, metrics in request["tool_usage"]["tool_breakdown"].items():
                if tool_name not in summary["tool_usage_stats"]:
                    summary["tool_usage_stats"][tool_name] = {
                        "total_uses": 0,
                        "successful_uses": 0,
                        "total_latency": 0.0
                    }
                
                summary["tool_usage_stats"][tool_name]["total_uses"] += metrics["count"]
                summary["tool_usage_stats"][tool_name]["successful_uses"] += metrics["success_count"]
                summary["tool_usage_stats"][tool_name]["total_latency"] += metrics["total_latency"]
            
            # Track latencies
            latencies.append(request["latency_metrics"]["total_latency"])
        
        # Calculate averages
        for metric, total in total_scores.items():
            summary["average_scores"][metric] = total / len(evaluated_requests)
        
        if latencies:
            summary["latency_stats"] = {
                "average": sum(latencies) / len(latencies),
                "min": min(latencies),
                "max": max(latencies)
            }
        
        return summary

@app.post("/evaluate")
async def evaluate_agent_responses(req: EvaluateRequest):
    """Evaluate all unevaluated responses for a specific agent."""
    try:
        # Validate agent name
        if req.agent_name not in ["learning_agent", "restaurant_agent"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent name. Must be one of: learning_agent, restaurant_agent"
            )
        
        # Create evaluator instance for the specified agent
        evaluator = AgentEvaluator(req.agent_name)
        
        # Start evaluation process
        evaluator.latency_tracker.start_action(
            ActionType.EVALUATION,
            {
                "action": "evaluate_unevaluated_requests",
                "agent_name": req.agent_name
            }
        )
        
        # Call the evaluation method
        evaluation_summary = evaluator.evaluate_unevaluated_requests()
        
        evaluator.latency_tracker.end_action(
            success=True,
            output=evaluation_summary
        )
        
        return {
            "status": "success",
            "message": f"Evaluation completed for {req.agent_name}",
            "summary": evaluation_summary
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.exception(f"Error during evaluation for {req.agent_name}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/evaluation/summary/{agent_name}")
async def get_agent_evaluation_summary(
    agent_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """Get a summary of all evaluations for a specific agent within a time period."""
    try:
        # Validate agent name
        if agent_name not in ["learning_agent", "restaurant_agent"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent name. Must be one of: learning_agent, restaurant_agent"
            )
        
        # Create evaluator instance for the specified agent
        evaluator = AgentEvaluator(agent_name)
        
        # Start getting summary
        evaluator.latency_tracker.start_action(
            ActionType.EVALUATION,
            {
                "action": "get_evaluation_summary",
                "agent_name": agent_name,
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
            "agent_name": agent_name,
            "summary": summary
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.exception(f"Error getting evaluation summary for {agent_name}")
        raise HTTPException(status_code=500, detail=error_msg)

# Create evaluator instances for both agents
learning_evaluator = AgentEvaluator("learning_agent")
restaurant_evaluator = AgentEvaluator("restaurant_agent")

# Export necessary components
__all__ = [
    'AgentEvaluator',
    'ActionType',
    'get_embedding',
    'learning_evaluator',
    'restaurant_evaluator'
]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Using port 8001 to avoid conflict with learning_agent 