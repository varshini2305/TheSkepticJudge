import yaml
import asyncio
import argparse
from pymongo import MongoClient
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from embedding_utils import update_existing_embeddings
from tqdm import tqdm
# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Initialize MongoDB
MONGO_URI = cfg["env"]["MONGODB_ATLAS_URI"]
LEARN_DB = cfg["env"]["mongodb"]["learning_agent"]["db"]
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[LEARN_DB]
materials_col = db["learning_materials"]

# Initialize both embedding models
genai.configure(api_key=cfg["env"]["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('embedding-001')
sentence_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

async def generate_gemini_embedding(text: str) -> list:
    """Generate embedding for text using Gemini."""
    try:
        result = await asyncio.to_thread(
            gemini_model.embed_content,
            text
        )
        return result.embedding
    except Exception as e:
        print(f"Error generating Gemini embedding: {str(e)}")
        raise

def generate_sentence_embedding(text: str, precision: str = "float32") -> list:
    """Generate embedding for text using SentenceTransformer."""
    try:
        embedding = sentence_model.encode(text, precision=precision).tolist()
        return embedding
    except Exception as e:
        print(f"Error generating SentenceTransformer embedding: {str(e)}")
        raise

async def update_document_embeddings(use_gemini: bool = True):
    """
    Update all documents with embeddings using either Gemini or SentenceTransformer.
    
    Args:
        use_gemini (bool): If True, use Gemini for embeddings, otherwise use SentenceTransformer
    """
    # Get all documents without embeddings
    cursor = materials_col.find({"embedding": {"$exists": False}})
    documents = list(cursor)  # Convert cursor to list
    print(f"Found {len(documents)} documents without embeddings")
    
    for doc in tqdm(documents):
        try:
            # Generate embedding for the document
            text = f"{doc.get('title', '')} {doc.get('course', '')} {doc.get('parsed_text', '')}"
            
            if use_gemini:
                embedding = await generate_gemini_embedding(text)
            else:
                embedding = await asyncio.to_thread(
                    generate_sentence_embedding,
                    text
                )
            
            # Update document with embedding
            materials_col.update_one(
                {"_id": doc["_id"]},
                {"$set": {"embedding": embedding}}
            )
            print(f"Updated document: {doc.get('title', 'Untitled')}")
            
        except Exception as e:
            print(f"Error updating document {doc.get('_id')}: {str(e)}")

async def main():
    parser = argparse.ArgumentParser(description='Update document embeddings using either Gemini or SentenceTransformer')
    parser.add_argument('--model', choices=['gemini', 'sentence'], default='gemini',
                      help='Choose embedding model: gemini or sentence (default: gemini)')
    
    args = parser.parse_args()
    use_gemini = args.model == 'gemini'
    
    print(f"Using {args.model} model for embeddings...")
    await update_document_embeddings(use_gemini)
    print("Finished updating embeddings!")

if __name__ == "__main__":
    asyncio.run(main())

# Update all documents without embeddings
asyncio.run(update_existing_embeddings()) 