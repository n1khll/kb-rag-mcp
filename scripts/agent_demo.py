"""
Agentic-RAG demo: an Azure AI Foundry agent (gpt-4o) that reaches out to the PUBLIC
kb-rag-mcp server over the internet, decides which tools to call, loops (search ->
assess confidence -> optionally refine or fetch full article), and answers with citations.

This is the "agentic" part: the loop lives in the model, driven by the instructions
below. The MCP server just exposes simple tools.

Usage (from kb_rag_mcp/ with .env populated):
  python scripts/agent_demo.py "my sap account is locked how do I reset the password"
"""
import os
import re
import sys
from pathlib import Path

from azure.identity import ClientSecretCredential
from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

AGENT_INSTRUCTIONS = (
    "You are a knowledge-base assistant. Answer strictly from the KB tools -- never from "
    "your own memory.\n"
    "1. Call `kb_search` with the user's question in their own words.\n"
    "2. Look at `confidence` / `low_confidence`. If low_confidence is true or the top results "
    "look off-topic, either refine the query and search again, or call `kb_list_categories` and "
    "re-search scoped to the right category.\n"
    "3. If a returned chunk looks right but seems truncated, call `kb_get_article` on its "
    "kb_number for the full text.\n"
    "4. Answer concisely, and ALWAYS cite the KB number(s) you used. If nothing relevant is "
    "found after searching, say so plainly -- do not invent an answer."
)


def main(question: str) -> None:
    tenant = os.environ["AZURE_TENANT_ID"]
    client_id = os.environ["AZURE_CLIENT_ID"]
    client_secret = os.environ["AZURE_CLIENT_SECRET"]
    project_endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
    model = os.environ["AZURE_AI_MODEL_DEPLOYMENT"]
    mcp_url = os.environ["MCP_PUBLIC_URL"]
    mcp_key = re.split(r",", os.environ["MCP_API_KEYS"])[0].strip()

    credential = ClientSecretCredential(tenant_id=tenant, client_id=client_id, client_secret=client_secret)
    project_client = AIProjectClient(endpoint=project_endpoint, credential=credential)
    openai_client = project_client.get_openai_client()

    tools = [
        {
            "type": "mcp",
            "server_label": "kb-rag",
            "server_url": mcp_url,
            "headers": {"Authorization": f"Bearer {mcp_key}"},
            "require_approval": "never",
        }
    ]

    print(f"QUESTION: {question}\n")
    response = openai_client.responses.create(
        model=model,
        input=question,
        instructions=AGENT_INSTRUCTIONS,
        tools=tools,
        tool_choice="auto",
    )

    # Show the agent's tool-call trace (proves the agentic loop) + final answer.
    print("--- agent tool-call trace ---")
    for item in getattr(response, "output", []) or []:
        item_type = getattr(item, "type", "")
        if item_type in ("mcp_call", "mcp_tool_call"):
            name = getattr(item, "name", "?")
            args = getattr(item, "arguments", "")
            print(f"  called {name}({str(args)[:120]})")
        elif item_type == "mcp_list_tools":
            print("  (discovered MCP tools)")

    print("\n--- final answer ---")
    print(getattr(response, "output_text", "") or "(no text)")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip() or "my sap account is locked, how do I reset the password"
    main(q)
