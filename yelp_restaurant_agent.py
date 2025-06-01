import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class YelpQueryInput(BaseModel):
    query: str = Field(description="The user's query for local business information")
    chat_id: str | None = Field(description="Unique chat ID for maintaining conversation history", default = None)


@tool(args_schema=YelpQueryInput)
def fusion_ai_api(query: str, chat_id: str) -> dict:
    """
    Calls Yelp Fusion AI API to get local business information and comparisons from a natural language query.

    Args:
    - query: The user's query for local business information.
    - chat_id: Unique chat ID for maintaining conversation history.

    Returns:
    - dict: JSON response from the Yelp Fusion AI API or an error message.
    """
    url = "https://api.yelp.com/ai/chat/v2"
    headers = {
      "Authorization": f"Bearer {YELP_API_KEY}",
      "Content-Type": "application/json"
    }
    data = {
      "query": query,
      "chat_id": chat_id
    }

    try:
      response = requests.post(url, headers=headers, json=data)
      response.raise_for_status()
      return response.json()
    except requests.RequestException as e:
      return {"error": f"API request failed: {str(e)}"}


from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4o")

# Define a system prompt for the agent
prompt = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful assistant"),
    ("human", "{input}"),
    # Placeholders fill up a **list** of messages
    ("placeholder", "{agent_scratchpad}"),
  ]
)

# Intialize the agent
agent = create_tool_calling_agent(llm, tools=[fusion_ai_api], prompt=prompt)
# Intialize AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=[fusion_ai_api], verbose=True)


input_data = {
  "query": "Find the best pizza places in SF",
  "chat_id": None  # None for first message
}

# Run the agent
response = agent_executor.invoke({"input": input_data})
print(response)