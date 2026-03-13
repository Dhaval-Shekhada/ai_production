import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "arcee-ai/trinity-large-preview:free"


class ChatRequest(BaseModel):
    prompt: str


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLM Tool</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; display: flex; justify-content: center; padding: 2rem; }
            .container { max-width: 640px; width: 100%; }
            h1 { margin-bottom: 1rem; }
            textarea { width: 100%; height: 100px; padding: 0.75rem; border-radius: 8px; border: 1px solid #333; background: #1a1a1a; color: #e0e0e0; font-size: 1rem; resize: vertical; }
            button { margin-top: 0.75rem; padding: 0.6rem 1.5rem; border: none; border-radius: 8px; background: #2563eb; color: #fff; font-size: 1rem; cursor: pointer; }
            button:disabled { opacity: 0.5; cursor: not-allowed; }
            #response { margin-top: 1.5rem; white-space: pre-wrap; background: #1a1a1a; padding: 1rem; border-radius: 8px; min-height: 60px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LLM Tool</h1>
            <textarea id="prompt" placeholder="Enter your prompt..."></textarea>
            <button id="send" onclick="send()">Send</button>
            <div id="response"></div>
        </div>
        <script>
            async function send() {
                const btn = document.getElementById('send');
                const out = document.getElementById('response');
                const prompt = document.getElementById('prompt').value.trim();
                if (!prompt) return;
                btn.disabled = true;
                out.textContent = 'Thinking...';
                try {
                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt })
                    });
                    const data = await res.json();
                    if (res.ok) {
                        out.textContent = data.reply;
                    } else {
                        out.textContent = 'Error: ' + (data.detail || JSON.stringify(data));
                    }
                } catch (e) {
                    out.textContent = 'Error: ' + e.message;
                } finally {
                    btn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/chat")
async def chat(req: ChatRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": req.prompt}],
            },
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="No choices returned from model")

    reply = choices[0]["message"]["content"]
    return {"reply": reply}