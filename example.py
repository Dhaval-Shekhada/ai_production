import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

# ---------------------------------------------------------------------------
# Provider registry — add/remove providers here only
# ---------------------------------------------------------------------------

PROVIDERS = {
    "openrouter": {
        "label": "OpenRouter (Trinity)",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "model": "arcee-ai/trinity-large-preview:free",
        "auth_header": lambda key: f"Bearer {key}",
    },
    "openai": {
        "label": "OpenAI (GPT-4o)",
        "url": "https://api.openai.com/v1/chat/completions",
        "key_env": "OPENAI_API_KEY",
        "model": "gpt-4o",
        "auth_header": lambda key: f"Bearer {key}",
    },
    "claude": {
        "label": "Anthropic (Claude 3.5 Sonnet)",
        "url": "https://api.anthropic.com/v1/messages",
        "key_env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-6",
        "auth_header": lambda key: key,  # Anthropic uses x-api-key, handled below
    },
    "gemini": {
        "label": "Google Gemini (1.5 Flash)",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-1.5-flash",
        "auth_header": lambda key: f"Bearer {key}",  # key passed as query param below
    },
}


def get_available_providers() -> list[dict]:
    """Return providers that have a non-empty API key configured."""
    available = []
    for provider_id, cfg in PROVIDERS.items():
        key = os.environ.get(cfg["key_env"], "").strip()
        available.append({
            "id": provider_id,
            "label": cfg["label"],
            "available": bool(key),
        })
    return available


async def call_provider(provider_id: str, prompt: str) -> str:
    """Call the specified provider and return the reply text."""
    cfg = PROVIDERS.get(provider_id)
    if not cfg:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_id}")

    key = os.environ.get(cfg["key_env"], "").strip()
    if not key:
        raise HTTPException(
            status_code=503,
            detail=f"API key not configured for provider '{cfg['label']}'. "
                   f"Set the {cfg['key_env']} environment variable.",
        )

    async with httpx.AsyncClient(timeout=60) as client:
        if provider_id == "claude":
            resp = await client.post(
                cfg["url"],
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": cfg["model"],
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
        elif provider_id == "gemini":
            resp = await client.post(
                f"{cfg['url']}?key={key}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
            )
        else:
            # OpenAI-compatible providers (OpenRouter, OpenAI)
            resp = await client.post(
                cfg["url"],
                headers={
                    "Authorization": cfg["auth_header"](key),
                    "Content-Type": "application/json",
                },
                json={
                    "model": cfg["model"],
                    "messages": [{"role": "user", "content": prompt}],
                },
            )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"[{cfg['label']}] {resp.text}",
        )

    data = resp.json()

    # Normalize response format across providers
    if provider_id == "claude":
        return data["content"][0]["text"]
    elif provider_id == "gemini":
        return data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Auto-fallback: try providers in order until one succeeds
# ---------------------------------------------------------------------------

async def call_any_available(prompt: str) -> tuple[str, str]:
    """Try all configured providers in order. Returns (reply, provider_label)."""
    errors = []
    for provider_id, cfg in PROVIDERS.items():
        key = os.environ.get(cfg["key_env"], "").strip()
        if not key:
            continue
        try:
            reply = await call_provider(provider_id, prompt)
            return reply, cfg["label"]
        except HTTPException as e:
            errors.append(f"{cfg['label']}: {e.detail}")

    raise HTTPException(
        status_code=503,
        detail="No working provider found. Errors: " + " | ".join(errors),
    )


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str
    provider: str = "auto"  # "auto" = try all until one works


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/providers")
async def list_providers():
    return {"providers": get_available_providers()}


@app.post("/chat")
async def chat(req: ChatRequest):
    if req.provider == "auto":
        reply, used_provider = await call_any_available(req.prompt)
        return {"reply": reply, "provider": used_provider}
    else:
        reply = await call_provider(req.provider, req.prompt)
        return {"reply": reply, "provider": PROVIDERS[req.provider]["label"]}


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
            select { width: 100%; padding: 0.6rem; border-radius: 8px; border: 1px solid #333; background: #1a1a1a; color: #e0e0e0; font-size: 1rem; margin-bottom: 0.75rem; }
            option:disabled { color: #555; }
            textarea { width: 100%; height: 100px; padding: 0.75rem; border-radius: 8px; border: 1px solid #333; background: #1a1a1a; color: #e0e0e0; font-size: 1rem; resize: vertical; }
            button { margin-top: 0.75rem; padding: 0.6rem 1.5rem; border: none; border-radius: 8px; background: #2563eb; color: #fff; font-size: 1rem; cursor: pointer; }
            button:disabled { opacity: 0.5; cursor: not-allowed; }
            #response { margin-top: 1.5rem; white-space: pre-wrap; background: #1a1a1a; padding: 1rem; border-radius: 8px; min-height: 60px; }
            .provider-tag { font-size: 0.75rem; color: #888; margin-top: 0.5rem; }
            .unavailable { color: #555; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>LLM Tool</h1>
            <select id="provider"><option value="auto">⚡ Auto (try all)</option></select>
            <textarea id="prompt" placeholder="Enter your prompt..."></textarea>
            <button id="send" onclick="send()">Send</button>
            <div id="response"></div>
            <div class="provider-tag" id="provider-tag"></div>
        </div>
        <script>
            async function loadProviders() {
                const sel = document.getElementById('provider');
                try {
                    const res = await fetch('/providers');
                    const data = await res.json();
                    data.providers.forEach(p => {
                        const opt = document.createElement('option');
                        opt.value = p.id;
                        opt.textContent = p.available ? p.label : p.label + ' (no key)';
                        opt.disabled = !p.available;
                        if (!p.available) opt.classList.add('unavailable');
                        sel.appendChild(opt);
                    });
                } catch (e) {
                    console.error('Failed to load providers', e);
                }
            }

            async function send() {
                const btn = document.getElementById('send');
                const out = document.getElementById('response');
                const tag = document.getElementById('provider-tag');
                const prompt = document.getElementById('prompt').value.trim();
                const provider = document.getElementById('provider').value;
                if (!prompt) return;
                btn.disabled = true;
                out.textContent = 'Thinking...';
                tag.textContent = '';
                try {
                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt, provider })
                    });
                    const data = await res.json();
                    if (res.ok) {
                        out.textContent = data.reply;
                        tag.textContent = 'Responded by: ' + data.provider;
                    } else {
                        out.textContent = 'Error: ' + (data.detail || JSON.stringify(data));
                    }
                } catch (e) {
                    out.textContent = 'Error: ' + e.message;
                } finally {
                    btn.disabled = false;
                }
            }

            loadProviders();
        </script>
    </body>
    </html>
    """
