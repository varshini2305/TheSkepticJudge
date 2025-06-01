import os
import yaml
import re
from pymongo import MongoClient
from datetime import datetime

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Initialize MongoDB client
MONGO_URI = cfg["env"]["MONGODB_ATLAS_URI"]
LEARN_DB = cfg["env"]["mongodb"]["learning_agent"]["db"]
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[LEARN_DB]
users_col = db["user_info"]

def generate_user_id(name: str, index: int) -> str:
    """Generate user ID in snake case with index."""
    # Convert to snake case
    name = name.lower()
    # Replace spaces and special characters with underscore
    name = re.sub(r'[^a-z0-9]+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Add index if greater than 1
    if index > 1:
        return f"{name}_{index}"
    return name

def update_existing_users():
    """Update existing users with user IDs."""
    # Get all users without user_id
    existing_users = list(users_col.find({"user_id": {"$exists": False}}))
    
    if not existing_users:
        print("No existing users found without user IDs")
        return
    
    # Track name occurrences
    name_counts = {}
    
    # Update existing users
    for user in existing_users:
        name = user["name"]
        name_counts[name] = name_counts.get(name, 0) + 1
        index = name_counts[name]
        
        # Generate user ID
        user_id = generate_user_id(name, index)
        
        # Update the user record
        try:
            users_col.update_one(
                {"_id": user["_id"]},
                {"$set": {"user_id": user_id}}
            )
            print(f"Updated user: {name} with ID: {user_id}")
        except Exception as e:
            print(f"Error updating user {name}: {str(e)}")

# Sample user data
users = [
    {"name": "Emma Thompson", "gender": "female", "interests": "reading, writing, poetry", "personality": "creative, introspective, analytical", "background": "enjoys deep discussions and learning through reading"},
    {"name": "James Wilson", "gender": "male", "interests": "sports, technology, gaming", "personality": "competitive, tech-savvy, strategic", "background": "learns best through hands-on experience and competition"},
    {"name": "Sophia Chen", "gender": "female", "interests": "music, dance, art", "personality": "artistic, expressive, detail-oriented", "background": "prefers visual and auditory learning methods"},
    {"name": "Lucas Rodriguez", "gender": "male", "interests": "cooking, photography, travel", "personality": "adventurous, creative, practical", "background": "learns through experience and experimentation"},
    {"name": "Olivia Parker", "gender": "female", "interests": "science, nature, hiking", "personality": "curious, observant, methodical", "background": "prefers learning through observation and experimentation"},
    {"name": "Ethan Kim", "gender": "male", "interests": "coding, robotics, puzzles", "personality": "logical, systematic, innovative", "background": "enjoys problem-solving and technical challenges"},
    {"name": "Ava Martinez", "gender": "female", "interests": "languages, culture, history", "personality": "open-minded, communicative, empathetic", "background": "learns best through cultural immersion"},
    {"name": "Noah Johnson", "gender": "male", "interests": "mathematics, chess, strategy games", "personality": "analytical, strategic, patient", "background": "prefers structured learning environments"},
    {"name": "Isabella Lee", "gender": "female", "interests": "medicine, biology, research", "personality": "dedicated, thorough, compassionate", "background": "enjoys scientific inquiry and research"},
    {"name": "William Brown", "gender": "male", "interests": "engineering, mechanics, DIY", "personality": "practical, innovative, hands-on", "background": "learns through building and experimentation"},
    {"name": "Mia Garcia", "gender": "female", "interests": "psychology, counseling, meditation", "personality": "empathetic, intuitive, supportive", "background": "prefers learning through interaction"},
    {"name": "Benjamin Taylor", "gender": "male", "interests": "finance, economics, investing", "personality": "analytical, strategic, detail-oriented", "background": "enjoys data-driven learning"},
    {"name": "Charlotte White", "gender": "female", "interests": "literature, theater, public speaking", "personality": "expressive, creative, confident", "background": "learns through performance and practice"},
    {"name": "Alexander Clark", "gender": "male", "interests": "physics, astronomy, technology", "personality": "inquisitive, logical, innovative", "background": "prefers theoretical and practical learning"},
    {"name": "Amelia Wright", "gender": "female", "interests": "education, child development, psychology", "personality": "nurturing, patient, organized", "background": "enjoys teaching and learning methods"},
    {"name": "Michael Anderson", "gender": "male", "interests": "sports, coaching, leadership", "personality": "motivational, strategic, team-oriented", "background": "learns through leadership and practice"},
    {"name": "Harper Moore", "gender": "female", "interests": "environmental science, sustainability, activism", "personality": "passionate, proactive, community-oriented", "background": "prefers hands-on environmental learning"},
    {"name": "Daniel Thomas", "gender": "male", "interests": "music production, sound engineering, technology", "personality": "creative, technical, detail-focused", "background": "learns through audio experimentation"},
    {"name": "Evelyn Jackson", "gender": "female", "interests": "fashion design, art history, cultural studies", "personality": "artistic, detail-oriented, culturally aware", "background": "enjoys visual and cultural learning"},
    {"name": "Matthew Harris", "gender": "male", "interests": "architecture, urban planning, design", "personality": "creative, analytical, visionary", "background": "prefers spatial and design learning"},
    {"name": "Abigail Martin", "gender": "female", "interests": "neuroscience, cognitive psychology, research", "personality": "analytical, curious, methodical", "background": "enjoys scientific research and analysis"},
    {"name": "David Thompson", "gender": "male", "interests": "cybersecurity, ethical hacking, programming", "personality": "technical, security-minded, problem-solver", "background": "learns through practical security challenges"},
    {"name": "Emily Davis", "gender": "female", "interests": "linguistics, translation, cultural studies", "personality": "precise, culturally sensitive, analytical", "background": "prefers language immersion learning"},
    {"name": "Joseph Wilson", "gender": "male", "interests": "quantum physics, mathematics, philosophy", "personality": "theoretical, philosophical, analytical", "background": "enjoys abstract and theoretical learning"},
    {"name": "Elizabeth Taylor", "gender": "female", "interests": "marine biology, oceanography, conservation", "personality": "observant, dedicated, environmentally conscious", "background": "learns through field research"},
    {"name": "Christopher Brown", "gender": "male", "interests": "artificial intelligence, machine learning, data science", "personality": "innovative, analytical, forward-thinking", "background": "prefers technical and mathematical learning"},
    {"name": "Sofia Garcia", "gender": "female", "interests": "international relations, diplomacy, languages", "personality": "diplomatic, culturally aware, communicative", "background": "enjoys cross-cultural learning"},
    {"name": "Andrew Lee", "gender": "male", "interests": "quantum computing, cryptography, mathematics", "personality": "technical, precise, innovative", "background": "prefers advanced technical learning"},
    {"name": "Victoria Chen", "gender": "female", "interests": "genetics, biotechnology, research", "personality": "analytical, innovative, detail-oriented", "background": "enjoys laboratory research"},
    {"name": "Ryan Kim", "gender": "male", "interests": "game development, computer graphics, animation", "personality": "creative, technical, detail-focused", "background": "learns through game development projects"}
]

def add_new_users():
    """Add new users to the database."""
    # Track name occurrences
    name_counts = {}
    
    # Add users and generate IDs
    for user in users:
        name = user["name"]
        name_counts[name] = name_counts.get(name, 0) + 1
        index = name_counts[name]
        
        # Generate user ID
        user_id = generate_user_id(name, index)
        
        # Add user ID to the record
        user["user_id"] = user_id
        user["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user["learning_history"] = []
        
        # Insert into database
        try:
            users_col.insert_one(user)
            print(f"Added user: {name} with ID: {user_id}")
        except Exception as e:
            print(f"Error adding user {name}: {str(e)}")

def main():
    print("Updating existing users with user IDs...")
    update_existing_users()
    
    print("\nAdding new users...")
    add_new_users()

if __name__ == "__main__":
    main() 