"""
Interactive agentic-RAG chat: type questions, watch the Foundry agent call the PUBLIC
kb-rag-mcp server, loop, and answer with citations. Type 'quit' to exit.

Usage (from kb_rag_mcp/ with .env populated):
  python scripts/chat.py
"""
import os
import re
from pathlib import Path

from azure.identity import ClientSecretCredential
from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

AGENT_INSTRUCTIONS = (
    "You are a knowledge-base assistant. Answer strictly from the KB tools -- never from "
    "your own memory.\n"
    "1. Call `kb_search` with the user's question in their own words.\n"
    "2. Check `confidence` / `low_confidence`. If low or off-topic, refine the query and "
    "search again, or use `kb_list_categories` and re-search scoped to a category.\n"
    "3. If a chunk looks right but truncated, call `kb_get_article` for the full text.\n"
    "4. Answer concisely and ALWAYS cite the KB number(s). If nothing relevant is found, "
    "say so plainly -- never invent an answer."
)


def main() -> None:
    credential = ClientSecretCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
    )
    project_client = AIProjectClient(endpoint=os.environ["AZURE_AI_PROJECT_ENDPOINT"], credential=credential)
    openai_client = project_client.get_openai_client()
    model = os.environ["AZURE_AI_MODEL_DEPLOYMENT"]
    mcp_key = re.split(r",", os.environ["MCP_API_KEYS"])[0].strip()
    tools = [
        {
            "type": "mcp",
            "server_label": "kb-rag",
            "server_url": os.environ["MCP_PUBLIC_URL"],
            "headers": {"Authorization": f"Bearer {mcp_key}"},
            "require_approval": "never",
        }
    ]

    print("Agentic KB chat -- type a question (or 'quit'). First call may take ~30-60s if the")
    print("server was asleep (Render free tier).\n")
    prev_id = None
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in {"quit", "exit"}:
            break

        kwargs = dict(model=model, input=question, instructions=AGENT_INSTRUCTIONS,
                      tools=tools, tool_choice="auto")
        if prev_id:
            kwargs["previous_response_id"] = prev_id
        response = openai_client.responses.create(**kwargs)
        prev_id = getattr(response, "id", None)

        called = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", "") in ("mcp_call", "mcp_tool_call"):
                called.append(getattr(item, "name", "?"))
        if called:
            print(f"   [tools used: {', '.join(called)}]")
        print(f"bot> {getattr(response, 'output_text', '') or '(no answer)'}\n")


if __name__ == "__main__":
    main()
