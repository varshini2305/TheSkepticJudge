import re
import json
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from openai import OpenAI
from datetime import datetime


# —————————— Load Config ——————————
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Initialize OpenAI client
openai_client = OpenAI(api_key=cfg["env"]["OPENAI_API_KEY"])

# Initialize MongoDB client
MONGO_URI = cfg["env"]["MONGODB_ATLAS_URI"]
REST_DB = cfg["env"]["mongodb"]["restaurant_agent"]["db"]
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[REST_DB]
users_col = db["user_info"]
rests_col = db["restaurant_info"]

# —————————— FastAPI Setup ——————————
app = FastAPI()

class RecommendRequest(BaseModel):
    user_name: str
    prompt: str

# —————————— Helpers ——————————

def get_user_profile(user_name: str):
    """Fetch a user document by their 'name' field."""
    return users_col.find_one({"name": user_name})

def extract_time_from_prompt(prompt: str):
    """
    Attempts to find a time in the prompt string.
    Returns a normalized 'HH:MM' (24h) string if found, else None.
    """
    prompt = prompt.lower()
    # Regex patterns for times like "7pm", "7:30pm", "19:00", "7:00 am", etc.
    patterns = [
        r"(\d{1,2}:\d{2}\s*(am|pm))",    # e.g. "7:30 pm"
        r"(\d{1,2}\s*(am|pm))",          # e.g. "7 pm"
        r"(\d{2}:\d{2})"                 # e.g. "19:00"
    ]
    for pat in patterns:
        m = re.search(pat, prompt)
        if m:
            raw = m.group(1).strip()
            try:
                # Parse with datetime.strptime (force lowercase)
                dt = None
                if re.search(r"(am|pm)", raw):
                    dt = datetime.strptime(raw, "%I:%M %p") if ":" in raw else datetime.strptime(raw, "%I %p")
                else:
                    dt = datetime.strptime(raw, "%H:%M")
                return dt.strftime("%-H:%M")  # e.g. "19:00" or "7:30"
            except:
                continue
    return None

def filter_restaurants(profile: dict, requested_time: str = None):
    """
    Return a list of restaurants with a 'remark' field explaining any mismatches.
    Each restaurant is scored by how many preferences it meets; higher scores come first.

    Preferences considered (soft):
      • city match
      • allergies accommodated
      • requested_time in open_slots
      • cuisine match
      • not previously visited

    Even if a restaurant fails one or more criteria, it is included with an appropriate remark.
    """
    desired_foods   = set(ft.lower() for ft in profile["preferences"]["food_type"])
    visited_names   = set(entry["restaurant_name"] for entry in profile.get("history", []))
    user_allergies  = set(a.lower() for a in profile["preferences"].get("allergies", []))
    user_city       = profile["location"]["city"].lower()

    scored = []
    for r in rests_col.find():
        reasons = []     # Collect reasons why this restaurant might be suboptimal
        score = 0        # Count of satisfied preferences

        # 1) City match
        rest_city = r["location"]["city"].lower()
        if rest_city == user_city:
            score += 1
        else:
            reasons.append(f"Located in {r['location']['city']}, outside your target city")

        # 2) Allergies
        rest_allergies = set(a.lower() for a in r.get("allergies_accommodated", []))
        missing_allergies = user_allergies - rest_allergies
        if not missing_allergies:
            score += 1
        else:
            miss_list = ", ".join(missing_allergies)
            reasons.append(f"May not accommodate allergy(ies): {miss_list}")

        # 3) Requested time availability
        if requested_time:
            normalized_slots = []
            for slot in r.get("open_slots", []):
                s = slot.strip().lower()
                try:
                    if re.search(r"(am|pm)", s):
                        dt = datetime.strptime(s, "%I:%M%p") if ":" in s else datetime.strptime(s, "%I%p")
                    else:
                        dt = datetime.strptime(s, "%H:%M")
                    normalized_slots.append(dt.strftime("%-H:%M"))
                except:
                    pass

            if requested_time in normalized_slots:
                score += 1
            else:
                reasons.append(f"No open slot at {requested_time}")
        else:
            # If no time requested, count as satisfied
            score += 1

        # 4) Cuisine match
        cuisine = r.get("cuisine", "").lower()
        if cuisine in desired_foods:
            score += 1
        else:
            reasons.append(f"Cuisine is '{r.get('cuisine', '')}', not in your preferred types")

        # 5) Visited before
        if r["name"] not in visited_names:
            score += 1
        else:
            reasons.append("You have visited this restaurant before")

        # Build remark
        if not reasons:
            remark = "Meets all your preferences"
        else:
            remark = "; ".join(reasons)

        # Attach remark and score to the restaurant data (avoid mutating original)
        restaurant_with_remark = r.copy()
        restaurant_with_remark["remark"] = remark
        scored.append((score, restaurant_with_remark))

    # Sort descending by score (higher score first). If tie, original order is preserved.
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return only the restaurant documents (with 'remark'), sorted by score
    return [restaurant for _, restaurant in scored]

def build_gpt4_prompt(user_prompt: str, profile: dict, candidates: list):
    """
    Construct a system prompt so GPT‑4 can pick one restaurant.
    """
    return (
        f"You are Restaurant Buddy. The user {profile['name']} asked: {user_prompt}\n"
        "User Profile:\n"
        f"- Name: {profile['name']}\n"
        f"- Dietary Preferences: {', '.join(profile['preferences']['food_type'])}\n"
        f"- Allergies: {', '.join(profile['preferences'].get('allergies', [])) or 'None'}\n"
        f"- Budget Range: {profile['preferences']['budget_range_usd']} USD\n"
        f"- Location Radius: {profile['preferences']['location_radius_km']} km\n"
        f"- Past Visits: {', '.join(entry['restaurant_name'] for entry in profile.get('history', [])) or 'None'}\n"
        f"- General Preference: {profile['preferences']['general_preference']}\n\n"
        "Candidate Restaurants (JSON array):\n"
        f"{json.dumps(candidates, indent=2, default=str)}\n\n"
        "Select exactly one restaurant that best fits the user's request (including time constraints). "
        "Output **only** a JSON object with these keys:\n"
        "  {\n"
        "    \"name\": <restaurant name>,\n"
        "    \"cuisine\": <cuisine>,\n"
        "    \"location\": <location object>,\n"
        "    \"contact_number\": <phone>,\n"
        "    \"website\": <website URL>,\n"
        "    \"ratings\": <ratings object>,\n"
        "    \"latest_reviews\": <array of review objects>,\n"
        "    \"explanation\": <brief explanation of your choice>\n"
        "  }\n"
        "Do not output anything else—only valid JSON."
    )

# —————————— Endpoint ——————————
@app.post("/recommend")
async def recommend(req: RecommendRequest):
    # 1) Fetch user profile
    profile = get_user_profile(req.user_name)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")

    # 2) Extract requested time (if any) from the prompt
    requested_time = extract_time_from_prompt(req.prompt)

    # 3) Filter restaurants based on profile + time
    candidates = filter_restaurants(profile, requested_time)
    if not candidates:
        raise HTTPException(status_code=404, detail="No matching restaurants found")

    # 4) Build GPT‑4 prompt and call the model
    system_prompt = build_gpt4_prompt(req.prompt, profile, candidates[:2])
    print(f"{system_prompt[:1000]=}")
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": system_prompt}],
        max_tokens=500,
        temperature=0.7
    )
    raw = response.choices[0].message.content.strip()

    # 5) Parse and return JSON
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM did not return valid JSON")

    return result