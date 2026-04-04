import os, json, logging
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as adk_types
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

async def fetch_wikipedia(topic: str) -> dict:
    try:
        clean = topic.replace("What is ","").replace("Tell me about ","").replace("Who is ","").replace("?","").strip()
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean.replace(' ','_')}"
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(url, headers={"User-Agent": "ADK-MCP-Agent/1.0"})
            if r.status_code == 200:
                d = r.json()
                return {"found": True, "title": d.get("title", topic), "summary": d.get("extract", "")[:500], "url": d.get("content_urls", {}).get("desktop", {}).get("page", "")}
            return {"found": False, "title": clean, "summary": f"No article found for {clean}.", "url": ""}
    except Exception as e:
        return {"found": False, "title": topic, "summary": str(e), "url": ""}

def wikipedia_mcp_tool(topic: str) -> dict:
    """MCP Tool: Retrieve factual information about any topic from Wikipedia. Use this whenever the user asks about a person, place, concept, or event. Args: topic: The subject to look up. Returns: A dict with title, summary, url."""
    import asyncio, concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, fetch_wikipedia(topic))
        return future.result()

SYSTEM_PROMPT = """
You are a helpful assistant that uses MCP to connect to Wikipedia.
When a user asks about any topic, ALWAYS call the wikipedia_mcp_tool first.
Then return ONLY this JSON format:
{
  "topic": "<topic searched>",
  "answer": "<clear answer based on Wikipedia data>",
  "source": "Wikipedia via MCP",
  "url": "<wikipedia url>",
  "model": "gemini"
}
"""

wiki_agent = Agent(name="wikipedia_mcp_agent", model=MODEL, description="ADK agent using MCP to fetch Wikipedia data.", instruction=SYSTEM_PROMPT, tools=[wikipedia_mcp_tool])
session_service = InMemorySessionService()
runner = Runner(agent=wiki_agent, app_name="adk-mcp-agent", session_service=session_service)
app = FastAPI(title="ADK Wikipedia MCP Agent")

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "mcp_tool": "wikipedia"}

@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    question = body.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="`question` field is required.")
    session = await session_service.create_session(app_name="adk-mcp-agent", user_id="user")
    content = adk_types.Content(role="user", parts=[adk_types.Part(text=question)])
    final_text = ""
    async for event in runner.run_async(user_id="user", session_id=session.id, new_message=content):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text.strip()
    try:
        result = json.loads(final_text.replace("```json","").replace("```","").strip())
    except:
        result = {"topic": question, "answer": final_text, "source": "Wikipedia via MCP", "url": "", "model": MODEL}
    return JSONResponse(content=result)

@app.get("/")
async def root():
    return {"agent": "ADK Wikipedia MCP Agent", "model": MODEL, "track": "Track 2"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
