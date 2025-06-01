import requests
import json
from datetime import datetime, timedelta
import time

def test_evaluation_endpoints():
    """Test the evaluation endpoints for the learning agent."""
    
    # Base URLs
    eval_base_url = "http://localhost:8001"
    
    # Test data
    agent_name = "learning_agent"
    end_time = datetime.now().isoformat()
    start_time = (datetime.now() - timedelta(hours=1)).isoformat()
    
    print("\n=== Testing Evaluation Endpoints ===\n")
    
    # 1. Test evaluate endpoint
    print("1. Testing /evaluate endpoint...")
    try:
        eval_response = requests.post(
            f"{eval_base_url}/evaluate",
            json={
                "agent_name": agent_name,
                "start_time": start_time,
                "end_time": end_time
            }
        )
        
        print(f"Status Code: {eval_response.status_code}")
        if eval_response.status_code == 200:
            print("Response:")
            print(json.dumps(eval_response.json(), indent=2))
        else:
            print(f"Error: {eval_response.text}")
            
    except Exception as e:
        print(f"Error testing evaluate endpoint: {str(e)}")
    
    print("\n2. Testing /evaluation/summary endpoint...")
    try:
        summary_response = requests.get(
            f"{eval_base_url}/evaluation/summary/{agent_name}",
            params={
                "start_time": start_time,
                "end_time": end_time
            }
        )
        
        print(f"Status Code: {summary_response.status_code}")
        if summary_response.status_code == 200:
            print("Response:")
            print(json.dumps(summary_response.json(), indent=2))
        else:
            print(f"Error: {summary_response.text}")
            
    except Exception as e:
        print(f"Error testing summary endpoint: {str(e)}")

if __name__ == "__main__":
    # Wait for services to start
    print("Waiting for services to start...")
    time.sleep(2)
    
    # Run tests
    test_evaluation_endpoints() 