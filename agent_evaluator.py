import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolUsage(BaseModel):
    tool_name: str
    timestamp: datetime
    input_params: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    response: Optional[Dict[str, Any]] = None

class AgentInteraction(BaseModel):
    session_id: str
    timestamp: datetime
    user_query: str
    tool_usages: List[ToolUsage]
    final_response: Dict[str, Any]
    response_quality: Optional[float] = None

class AgentEvaluator:
    def __init__(self):
        self.interactions: List[AgentInteraction] = []
        
    def log_tool_usage(self, tool_name: str, input_params: Dict[str, Any], 
                      success: bool, response: Optional[Dict[str, Any]] = None,
                      error_message: Optional[str] = None) -> ToolUsage:
        """Log a tool usage event."""
        tool_usage = ToolUsage(
            tool_name=tool_name,
            timestamp=datetime.now(),
            input_params=input_params,
            success=success,
            response=response,
            error_message=error_message
        )
        logger.info(f"Tool Usage: {tool_usage.dict()}")
        return tool_usage
    
    def log_interaction(self, session_id: str, user_query: str, 
                       tool_usages: List[ToolUsage], final_response: Dict[str, Any]):
        """Log a complete agent interaction."""
        interaction = AgentInteraction(
            session_id=session_id,
            timestamp=datetime.now(),
            user_query=user_query,
            tool_usages=tool_usages,
            final_response=final_response
        )
        self.interactions.append(interaction)
        logger.info(f"Interaction logged: {interaction.dict()}")
        
    def evaluate_tool_usage_patterns(self) -> Dict[str, Any]:
        """Evaluate patterns in tool usage."""
        tool_stats = {}
        for interaction in self.interactions:
            for tool_usage in interaction.tool_usages:
                if tool_usage.tool_name not in tool_stats:
                    tool_stats[tool_usage.tool_name] = {
                        "total_calls": 0,
                        "successful_calls": 0,
                        "failed_calls": 0,
                        "avg_response_time": 0
                    }
                
                stats = tool_stats[tool_usage.tool_name]
                stats["total_calls"] += 1
                if tool_usage.success:
                    stats["successful_calls"] += 1
                else:
                    stats["failed_calls"] += 1
                    
        return tool_stats
    
    def evaluate_response_quality(self, interaction: AgentInteraction) -> float:
        """
        Evaluate response quality based on various factors:
        1. Tool usage appropriateness
        2. Response completeness
        3. Error handling
        """
        score = 0.0
        max_score = 100.0
        
        # Check if appropriate tools were used
        if interaction.tool_usages:
            tool_score = 0.0
            for tool_usage in interaction.tool_usages:
                if tool_usage.success:
                    tool_score += 20.0  # Each successful tool usage adds points
            score += min(tool_score, 40.0)  # Max 40 points for tool usage
        
        # Check response completeness
        if interaction.final_response:
            # Check if response has required fields
            required_fields = ["recommendations", "summary"]  # Adjust based on agent type
            completeness_score = 0.0
            for field in required_fields:
                if field in interaction.final_response:
                    completeness_score += 20.0
            score += min(completeness_score, 40.0)  # Max 40 points for completeness
        
        # Check error handling
        error_score = 20.0
        for tool_usage in interaction.tool_usages:
            if not tool_usage.success and not tool_usage.error_message:
                error_score -= 5.0  # Deduct points for missing error messages
        score += max(error_score, 0.0)  # Max 20 points for error handling
        
        return score
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        report = {
            "total_interactions": len(self.interactions),
            "tool_usage_stats": self.evaluate_tool_usage_patterns(),
            "average_response_quality": 0.0,
            "interaction_details": []
        }
        
        total_quality = 0.0
        for interaction in self.interactions:
            quality = self.evaluate_response_quality(interaction)
            total_quality += quality
            
            report["interaction_details"].append({
                "session_id": interaction.session_id,
                "timestamp": interaction.timestamp.isoformat(),
                "user_query": interaction.user_query,
                "response_quality": quality,
                "tool_usages": [usage.dict() for usage in interaction.tool_usages]
            })
        
        if self.interactions:
            report["average_response_quality"] = total_quality / len(self.interactions)
            
        return report
    
    def save_evaluation_report(self, filename: str):
        """Save the evaluation report to a file."""
        report = self.generate_evaluation_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Evaluation report saved to {filename}")

# Create a singleton instance
evaluator = AgentEvaluator() 