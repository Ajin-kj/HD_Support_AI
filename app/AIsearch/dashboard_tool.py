from langchain_core.tools import StructuredTool
import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_dashboard_html(ticket_data: str) -> dict:
    """
    Analyzes ticket data and generates an HTML dashboard with insights and charts.
    Expects JSON-formatted ticket_data string.
    Returns: {
        "message": "Here is your dashboard...",
        "html_code": "<html>...</html>"
    }
    """
    # You can customize this prompt further based on your use case
    prompt_template = PromptTemplate.from_template("""
    You are a business analyst. Given the following ticket data in JSON format, 
    analyze and create insights. Create an HTML dashboard with charts, tables, 
    and tiles that help the user understand issues, trends, or KPIs.
    
    Data: {ticket_data}
    
    Respond with a summary message first, then a valid HTML code for the dashboard.
    Use simple charts (bar, pie, table) using any popular charting lib (e.g. Chart.js).
    """)

    prompt = prompt_template.format(ticket_data=ticket_data)
    response = llm.invoke(prompt)

    # Basic extraction (assuming split between message and code with delimiter)
    if "<html" in response:
        split_index = response.index("<html")
        message = response[:split_index].strip()
        html_code = response[split_index:]
    else:
        message = "Dashboard generated."
        html_code = response

    return {"message": message, "html_code": html_code}

# Wrap as structured tool
dashboard_tool_structured = StructuredTool.from_function(
    name="generate_dashboard",
    func=generate_dashboard_html,
    description="Generate an HTML dashboard from ticket data. Input is ticket data in JSON format. Output is a summary and HTML dashboard code."
)
