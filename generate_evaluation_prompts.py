import os
import json
import logging
import re
from typing import Dict, Set
from langchain_openai import ChatOpenAI
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Initialize OpenAI model
eval_llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    api_key=cfg["env"]["OPENAI_API_KEY"]
)

def clean_json_response(response_text: str) -> str:
    """Clean the response text to ensure valid JSON."""
    # Remove any markdown code block markers
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*$', '', response_text)
    
    # Remove any control characters
    response_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', response_text)
    
    # Ensure the response is properly formatted JSON
    try:
        # Try to parse and re-serialize to ensure valid JSON
        parsed = json.loads(response_text)
        return json.dumps(parsed)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        logger.error(f"Problematic response text: {response_text}")
        raise

def generate_agent_prompts(agent_name: str, task_description: str) -> Dict[str, Dict[str, str]]:
    """Generate evaluation prompts for a specific agent type."""
    
    # Define the metaprompt for generating evaluation criteria
    metaprompt = """You are a prompt engineering expert tasked with creating evaluation criteria for an AI assistant.
    The agent's task is: {task_description}
    
    For each evaluation metric, create a detailed prompt that will guide the evaluation of the agent's responses.
    Each prompt should be specific to the agent's domain and purpose.
    
    Return a JSON object with the following structure and detailed prompts:
    {{
        "clarity": {{
            "prompt": "Evaluate how clear and understandable the response is. Consider:
            1. Is the language simple and accessible?
            2. Are complex concepts or terms explained clearly?
            3. Is the information presented in a structured way?
            4. Are examples or explanations concrete and relatable?
            5. Is the response free from unnecessary jargon?
            
            Score from 0 to 1, where:
            0 = Confusing, uses complex language, or lacks clear explanations
            0.5 = Somewhat clear but could be simplified
            1 = Crystal clear with appropriate language and explanations
            
            Return only the score as a float.",
            "description": "Clarity means making information accessible and understandable to users."
        }},
        "completeness": {{
            "prompt": "Evaluate how thoroughly the response addresses the user's needs. Consider:
            1. Does it cover all necessary aspects of the request?
            2. Does it provide appropriate context and details?
            3. Does it address potential concerns or questions?
            4. Does it include relevant examples or options?
            5. Does it connect the information to user's specific needs?
            
            Score from 0 to 1, where:
            0 = Missing critical information or aspects
            0.5 = Covers main points but lacks depth or details
            1 = Comprehensive coverage with appropriate details and connections
            
            Return only the score as a float.",
            "description": "Completeness means providing comprehensive coverage of the user's needs while maintaining clarity."
        }},
        "relevance": {{
            "prompt": "Evaluate how relevant and focused the response is to the user's request. Consider:
            1. Does it directly address the user's specific query?
            2. Is the level of detail appropriate for the request?
            3. Are examples and options relevant to the user's needs?
            4. Does it stay focused on the key requirements?
            5. Does it avoid unnecessary information or tangents?
            
            Score from 0 to 1, where:
            0 = Off-topic or inappropriate level of detail
            0.5 = Somewhat relevant but could be more focused
            1 = Perfectly targeted to the user's needs and query
            
            Return only the score as a float.",
            "description": "Relevance means providing information that is directly applicable to the user's current needs."
        }},
        "engagement": {{
            "prompt": "Evaluate how engaging and user-friendly the response is. Consider:
            1. Does it maintain user interest through appropriate pacing?
            2. Does it use engaging examples or options?
            3. Does it encourage user interaction or feedback?
            4. Is the tone appropriate and welcoming?
            5. Does it make the information appealing and accessible?
            
            Score from 0 to 1, where:
            0 = Passive, dry response with no engagement
            0.5 = Some engaging elements but could be more interactive
            1 = Highly engaging with user-friendly elements
            
            Return only the score as a float.",
            "description": "Engagement means actively involving users and maintaining their interest in the interaction."
        }},
        "instruction_following_quality": {{
            "prompt": "Evaluate how well the response adheres to the specific requirements and constraints. Consider:
            1. Does it follow all specified guidelines and protocols?
            2. Does it respect user preferences and constraints?
            3. Does it maintain appropriate boundaries and limitations?
            4. Does it handle special requirements appropriately?
            5. Does it follow all necessary procedures?
            
            Score from 0 to 1, where:
            0 = Violates constraints or guidelines
            0.5 = Partially follows requirements
            1 = Perfectly follows all requirements and constraints
            
            Return only the score as a float.",
            "description": "Instruction following quality means adhering to specific requirements and constraints while maintaining effectiveness."
        }}
    }}
    
    Make the prompts specific to the agent's domain and purpose. Consider:
    1. How well does the response serve the user's needs?
    2. Does it maintain appropriate domain-specific standards?
    3. Is it appropriate for the user's context?
    4. Does it follow best practices for the domain?
    5. Does it maintain user satisfaction and trust?
    
    Return ONLY the JSON object, with no additional text or explanation."""
    
    try:
        # Generate prompts using the metaprompt
        response = eval_llm.invoke(metaprompt.format(task_description=task_description))
        
        # Clean and parse the response
        cleaned_response = clean_json_response(response.content)
        prompts = json.loads(cleaned_response)
        
        # Save prompts to file
        prompts_dir = "data/evaluation_prompts"
        os.makedirs(prompts_dir, exist_ok=True)
        
        output_file = os.path.join(prompts_dir, f"{agent_name}_prompts.json")
        with open(output_file, "w") as f:
            json.dump(prompts, f, indent=2)
        
        logger.info(f"Generated evaluation prompts for {agent_name} and saved to {output_file}")
        return prompts
        
    except Exception as e:
        logger.error(f"Error generating prompts for {agent_name}: {str(e)}")
        raise

if __name__ == "__main__":
    # Task descriptions for different agents
    agent_tasks = {
        "learning_agent": """This agent assists students in learning topics from their study materials.
        It uses effective teaching techniques to:
        - Keep answers clear and easy to understand
        - Personalize responses when relevant
        - Make learning engaging
        - Use Socratic method to prompt students to think
        - Avoid giving answers away too easily
        - Guide students to discover solutions themselves
        
        The agent should evaluate how well responses achieve these teaching goals.""",
        
        "restaurant_agent": """This agent provides personalized restaurant recommendations based on user requests.
        It automatically considers:
        - User's general preferences and restaurant visit history (to avoid repetitions)
        - Budget range and constraints
        - Dietary restrictions and allergies
        - Cuisine preferences
        - Any specific requirements mentioned in the user's query
        
        The agent should:
        - Provide well-reasoned recommendations
        - Explain why each recommendation matches the user's needs
        - Consider multiple factors in its suggestions
        - Maintain a helpful and engaging tone
        - Handle special requirements appropriately
        
        The agent should evaluate how well responses meet these recommendation goals."""
    }
    
    # Generate prompts for each agent
    for agent_name, task_description in agent_tasks.items():
        logger.info(f"Generating prompts for {agent_name}...")
        try:
            generate_agent_prompts(agent_name, task_description)
        except Exception as e:
            logger.error(f"Failed to generate prompts for {agent_name}: {str(e)}")
            continue 