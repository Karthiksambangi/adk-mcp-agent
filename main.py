import os, json, logging
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from groq import Groq

logging.basicConfig(level=logging.INFO)
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

async def fetch_wikipedia(topic: str) -> dict:
    try:
        clean_topic = topic.replace("What is ", "").replace("Tell me about ", "").replace("Who is ", "").replace("?", "").strip()
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_topic.replace(' ', '_')}"
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(url, headers={"User-Agent": "ADK-MCP-Agent/1.0"})
            if r.status_code == 200:
                d = r.json()
                return {"found": True, "title": d.get("title", topic), "summary": d.get("extract", "")[:500], "url": d.get("content_urls", {}).get("desktop", {}).get("page", "")}
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_topic.split()[0]}"
            r2 = await c.get(search_url, headers={"User-Agent": "ADK-MCP-Agent/1.0"})
            if r2.status_code == 200:
                d = r2.json()
                return {"found": True, "title": d.get("title", topic), "summary": d.get("extract", "")[:500], "url": d.get("content_urls", {}).get("desktop", {}).get("page", "")}
            return {"found": False, "title": clean_topic, "summary": f"Could not find Wikipedia article for {clean_topic}.", "url": ""}
    except Exception as e:
        return {"found": False, "title": topic, "summary": str(e), "url": ""}

app = FastAPI(title="ADK Wikipedia MCP Agent")

@app.get("/health")
async def health():
    return {"status": "ok", "model": "llama-3.3-70b", "mcp_tool": "wikipedia"}

@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    question = body.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="`question` field is required.")
    wiki_data = await fetch_wikipedia(question)
    prompt = f"""You are a helpful assistant. Use this Wikipedia data to answer the question.
Wikipedia data: {json.dumps(wiki_data)}
Question: {question}
Return ONLY this JSON with no extra text:
{{"topic": "{wiki_data.get('title', question)}", "answer": "your detailed answer here based on wikipedia data", "source": "Wikipedia via MCP", "url": "{wiki_data.get('url', '')}", "model": "llama"}}"""
    try:
        response = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": prompt}], temperature=0.3)
        result_text = response.choices[0].message.content.strip()
        cleaned = result_text.replace("```json","").replace("```","").strip()
        result = json.loads(cleaned)
    except Exception as e:
        result = {"topic": wiki_data.get("title", question), "answer": wiki_data.get("summary", str(e)), "source": "Wikipedia via MCP", "url": wiki_data.get("url", ""), "model": "llama"}
    return JSONResponse(content=result)

@app.get("/")
async def root():
    return {"agent": "ADK Wikipedia MCP Agent", "model": "llama-3.3-70b", "track": "Track 2"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
