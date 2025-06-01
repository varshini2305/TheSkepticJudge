import os
import yaml
import re
from pymongo import MongoClient
from collections import defaultdict

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Initialize MongoDB
MONGO_URI = cfg["env"]["MONGODB_ATLAS_URI"]
RESTAURANT_DB = cfg["env"]["mongodb"]["restaurant_agent"]["db"]
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[RESTAURANT_DB]
restaurants_col = db["restaurant_info"]
users_col = db["user_info"]

def generate_restaurant_id(name: str, pincode: str) -> str:
    """Generate restaurant ID from name and pincode."""
    # Remove any leading/trailing whitespace
    name = name.strip()
    
    # Find the first word by splitting on any non-alphanumeric character
    # This includes spaces, apostrophes, hyphens, etc.
    first_word = re.split(r'[^a-zA-Z0-9]', name)[0]
    
    # Convert to uppercase
    first_word = first_word.upper()
    
    # Combine with pincode
    return f"{first_word}{pincode}"

def generate_user_id(name: str, index: int = None) -> str:
    """Generate user ID from name with optional index."""
    # Convert to snake case
    snake_case = re.sub(r'[\s\-]+', '_', name.lower())
    # Remove special characters
    snake_case = re.sub(r'[^a-z0-9_]', '', snake_case)
    # Add index if provided
    if index is not None:
        return f"{snake_case}_{index}"
    return snake_case

def update_restaurant_ids():
    """Update restaurant IDs in the restaurant_info collection."""
    print("Updating restaurant IDs...")
    
    # Get all restaurants
    restaurants = list(restaurants_col.find({}))
    print(f"Found {len(restaurants)} restaurants")
    
    # Update each restaurant
    for restaurant in restaurants:
        try:
            name = restaurant.get("name", "")
            pincode = restaurant.get("location", {}).get("pincode", "")
            
            if name and pincode:
                restaurant_id = generate_restaurant_id(name, pincode)
                
                # Update the document
                restaurants_col.update_one(
                    {"_id": restaurant["_id"]},
                    {"$set": {"restaurant_id": restaurant_id}}
                )
                print(f"Updated restaurant: {name} -> {restaurant_id}")
            else:
                print(f"Skipping restaurant {restaurant['_id']}: Missing name or pincode")
                
        except Exception as e:
            print(f"Error updating restaurant {restaurant['_id']}: {str(e)}")
    
    print("Restaurant ID update completed")

def update_user_ids():
    """Update user IDs in the user_info collection."""
    print("Updating user IDs...")
    
    # Get all users
    users = list(users_col.find({}))
    print(f"Found {len(users)} users")
    
    # Count occurrences of each name
    name_counts = defaultdict(int)
    for user in users:
        name = user.get("name", "")
        if name:
            name_counts[name] += 1
    
    # Update each user
    for user in users:
        try:
            name = user.get("name", "")
            if name:
                # If name appears multiple times, add index
                if name_counts[name] > 1:
                    # Get current count and increment
                    current_count = name_counts[name]
                    name_counts[name] -= 1
                    user_id = generate_user_id(name, current_count)
                else:
                    user_id = generate_user_id(name)
                
                # Update the document
                users_col.update_one(
                    {"_id": user["_id"]},
                    {"$set": {"user_id": user_id}}
                )
                print(f"Updated user: {name} -> {user_id}")
            else:
                print(f"Skipping user {user['_id']}: Missing name")
                
        except Exception as e:
            print(f"Error updating user {user['_id']}: {str(e)}")
    
    print("User ID update completed")

if __name__ == "__main__":
    try:
        # Update restaurant IDs
        update_restaurant_ids()
        
        # Update user IDs
        update_user_ids()
        
        print("All updates completed successfully")
        
    except Exception as e:
        print(f"Error during update process: {str(e)}")
    finally:
        mongo_client.close() 