from typing import List, Dict
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import StructuredTool
from app.AIsearch.Metrics_service import search_metrics
from app.AIsearch.Support_service import search_support
from app.AIsearch.Ticket_service import search_tickets  # <-- New ticket vector search
from app.AIsearch.dashboard_tool import dashboard_tool_structured
from app.config import Config
import json

# === Simple in-memory session store ===
SESSION_STORE: Dict[str, List] = {}

# === Initialize Azure LLM ===
llm = AzureChatOpenAI(
    azure_endpoint=Config.AI_AZURE_OPENAI_ENDPOINT,
    api_key=Config.AI_AZURE_OPENAI_API_KEY,
    api_version=Config.AI_API_VERSION,
    deployment_name=Config.AI_DEPLOYMENT_NAME,
    model_name=Config.AI_MODEL_NAME,
    temperature=0.5,
)

# === Internal Tool Wrappers ===
def support_tool(query: str, top_k: int = 1) -> str:
    results = search_support(query, top_k)
    return json.dumps(results, indent=2) if results else "No relevant support information found."

def metrics_tool(query: str, top_k: int = 1) -> str:
    results = search_metrics(query, top_k)
    return json.dumps(results, indent=2) if results else "No relevant metrics found."

def ticket_tool(query: str, top_k: int = 1) -> str:
    results = search_tickets(query, top_k)
    return json.dumps(results, indent=2) if results else "No relevant ticket data found."

# === Structured Tools ===
support_tool_structured = StructuredTool.from_function(
    name="search_support",
    func=support_tool,
    description="Search support issues given a query. Use for logs, errors, and tech issues."
)

metrics_tool_structured = StructuredTool.from_function(
    name="search_metrics",
    func=metrics_tool,
    description="Search for performance metrics based on user query, such as dates, countries, reports."
)

ticket_tool_structured = StructuredTool.from_function(
    name="search_tickets",
    func=ticket_tool,
    description="Search ticket summaries based on user query. Use this for queries mentioning incidents, ticket summaries, ticket issues, or resolutions. Always keep k=1"
)



# === Register All Tools ===
tools = [support_tool_structured, metrics_tool_structured, ticket_tool_structured,]

# === Prompt Template ===
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are Nitrous, an AI assistant that calls internal tools for support, performance metrics, or ticket summaries.\n"
"Follow these rules:\n"
"- Use `search_metrics` for performance/ report/ metrics queries with dates/locations (use top_k=1 or 2).\n"
"- In metrics query, replace 'performance report' with 'performance metrics' for better search results.\n"
"- Always ask for a date if the query relates to performance report `search_metrics` but lacks one.\n"
"- Use `search_support` for logs, issues, or technical problems. Use top_k=3+ if vague.\n"
"- Use `search_tickets` for queries that mention 'tickets', 'incidents', 'resolutions', or ticket logs.\n"
"- If the user asks about an 'issue' and it's unclear whether it's support or ticket related, ask them to clarify.\n"
"- If a support query is vague, ask for clarification or list known issues.\n"
"- When a user says 'hi' or similar greetings, respond with: 'Hi, I'm Nitrous, your AI assistant. How can I help you today?'\n"
"- You must not answer any other questions from your own knowledge. Only respond based on results from the internal search tools.\n"
"- If a user asks something outside the knowledge base or Azure AI Search-provided data, politely respond: 'I'm sorry, I can only assist with information available through the H-D knowledge base. Please try rephrasing your query related to support, performance metrics, or tickets. If you need further help, please contact your administrator.'\n"

     ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# === Agent Setup ===
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# === Main Function to Run Agent Query ===
def run_agent_query(user_input: str, session_id: str):
    # Get or initialize chat history
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = []

    chat_history = SESSION_STORE[session_id]

    # Run the agent
    response = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    # Store current round of conversation
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.get("output", "No response generated.")))

    # Store back in session
    SESSION_STORE[session_id] = chat_history

    # Parse steps
    steps = []
    for action, observation in response.get("intermediate_steps", []):
        steps.append({
            "tool_used": action.tool,
            "tool_input": action.tool_input,
            "result": observation
        })

    return {
        "session_id": session_id,
        "steps": steps,
        "final_output": response.get("output", "No response generated.")
    }
