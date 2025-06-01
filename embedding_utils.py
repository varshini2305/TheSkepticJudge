import yaml
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import asyncio
from typing import Dict, Any

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Initialize MongoDB
MONGO_URI = cfg["env"]["MONGODB_ATLAS_URI"]
LEARN_DB = cfg["env"]["mongodb"]["learning_agent"]["db"]
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[LEARN_DB]
materials_col = db["learning_materials"]

# Initialize the embedding model
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

def generate_embedding(text: str, precision: str = "float32") -> list:
    """
    Generate vector embedding for a given text using the sentence transformer model.
    
    Args:
        text (str): The text to generate embedding for
        precision (str): The precision of the embedding (default: "float32")
    
    Returns:
        list: The vector embedding as a list of floats
    """
    try:
        embedding = model.encode(text, precision=precision).tolist()
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise

async def insert_document_with_embedding(document: Dict[str, Any]) -> str:
    """
    Insert a document into the learning_materials collection with its vector embedding.
    
    Args:
        document (Dict[str, Any]): The document to insert, must contain 'parsed_text' field
    
    Returns:
        str: The ID of the inserted document
    """
    try:
        # Generate embedding for the parsed_text
        if "parsed_text" not in document:
            raise ValueError("Document must contain 'parsed_text' field")
            
        # Generate embedding asynchronously to not block the event loop
        embedding = await asyncio.to_thread(
            generate_embedding,
            document["parsed_text"]
        )
        
        # Add the embedding to the document
        document["embedding"] = embedding
        
        # Insert the document
        result = materials_col.insert_one(document)
        print(f"Inserted document with ID: {result.inserted_id}")
        
        return str(result.inserted_id)
        
    except Exception as e:
        print(f"Error inserting document: {str(e)}")
        raise

async def update_existing_embeddings():
    """
    Update embeddings for all documents in the collection that don't have embeddings.
    """
    # Find all documents without embeddings
    documents = materials_col.find({"embedding": {"$exists": False}})
    
    for doc in documents:
        try:
            if "parsed_text" not in doc:
                print(f"Skipping document {doc.get('_id')}: No parsed_text field")
                continue
                
            # Generate embedding
            embedding = await asyncio.to_thread(
                generate_embedding,
                doc["parsed_text"]
            )
            
            # Update document with embedding
            materials_col.update_one(
                {"_id": doc["_id"]},
                {"$set": {"embedding": embedding}}
            )
            print(f"Updated document: {doc.get('title', 'Untitled')}")
            
        except Exception as e:
            print(f"Error updating document {doc.get('_id')}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    async def main():
        # Example document
        doc = {
            "title": "Example Document",
            "course": "Example Course",
            "parsed_text": "This is an example document for testing embeddings."
        }
        
        # Insert document with embedding
        doc_id = await insert_document_with_embedding(doc)
        print(f"Inserted document with ID: {doc_id}")
        
        # Update existing documents without embeddings
        await update_existing_embeddings()

    asyncio.run(main()) 