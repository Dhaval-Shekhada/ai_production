import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

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
    "ollama": {
        "label": "Ollama (Qwen 3.5 9B)",
        "url": "http://127.0.0.1:11434/v1/chat/completions",
        "key_env": None,  # local Ollama does not require an API key
        "model": "qwen3.5:9b",
        "auth_header": lambda key: "Bearer ollama",
        "timeout": 180,
    },
}

OLLAMA_CLIENT_URL = "http://127.0.0.1:11434/v1/chat/completions"


def _ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", "").strip() or PROVIDERS["ollama"]["url"]


def _is_loopback(url: str) -> bool:
    return "localhost" in url or "127.0.0.1" in url


def _server_can_call_ollama() -> bool:
    """Vercel cannot reach the user's localhost Ollama unless a public tunnel URL is set."""
    if os.environ.get("VERCEL") and _is_loopback(_ollama_url()):
        return False
    return True


def _provider_key(cfg: dict) -> str:
    env_name = cfg.get("key_env")
    if not env_name:
        return ""
    return os.environ.get(env_name, "").strip()


def _provider_available(cfg: dict) -> bool:
    if not cfg.get("key_env"):
        return True
    return bool(_provider_key(cfg))


def get_available_providers() -> list[dict]:
    """Return providers and whether they are ready to use."""
    available = []
    for provider_id, cfg in PROVIDERS.items():
        available.append({
            "id": provider_id,
            "label": cfg["label"],
            "available": _provider_available(cfg),
            "client_local": provider_id == "ollama" and not _server_can_call_ollama(),
        })
    return available


def _openai_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        role = "assistant" if m.get("role") == "assistant" else "user"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out


def _claude_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in _openai_messages(messages):
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"] += "\n" + m["content"]
        else:
            out.append(m)
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "(continue)"})
    return out


def _gemini_contents(messages: list[dict]) -> list[dict]:
    contents = []
    for m in _openai_messages(messages):
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    return contents


async def call_provider(provider_id: str, messages: list[dict]) -> str:
    """Call the specified provider and return the reply text."""
    cfg = PROVIDERS.get(provider_id)
    if not cfg:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_id}")

    key = _provider_key(cfg)
    if cfg.get("key_env") and not key:
        raise HTTPException(
            status_code=503,
            detail=f"API key not configured for provider '{cfg['label']}'. "
                   f"Set the {cfg['key_env']} environment variable.",
        )

    openai_msgs = _openai_messages(messages)
    if not openai_msgs:
        raise HTTPException(status_code=400, detail="No messages to send.")

    timeout = cfg.get("timeout", 60)
    request_url = _ollama_url() if provider_id == "ollama" else cfg["url"]
    async with httpx.AsyncClient(timeout=timeout) as client:
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
                    "messages": _claude_messages(messages),
                },
            )
        elif provider_id == "gemini":
            resp = await client.post(
                f"{cfg['url']}?key={key}",
                headers={"Content-Type": "application/json"},
                json={"contents": _gemini_contents(messages)},
            )
        else:
            # OpenAI-compatible providers (OpenRouter, OpenAI, Ollama)
            resp = await client.post(
                request_url,
                headers={
                    "Authorization": cfg["auth_header"](key),
                    "Content-Type": "application/json",
                },
                json={
                    "model": cfg["model"],
                    "messages": openai_msgs,
                },
            )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"[{cfg['label']}] {resp.text}",
        )

    data = resp.json()

    if provider_id == "claude":
        return data["content"][0]["text"]
    elif provider_id == "gemini":
        return data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Auto-fallback: try providers in order until one succeeds
# ---------------------------------------------------------------------------

async def call_any_available(messages: list[dict]) -> tuple[str, str]:
    """Try all configured providers in order. Returns (reply, provider_label)."""
    errors = []
    for provider_id, cfg in PROVIDERS.items():
        if not _provider_available(cfg):
            continue
        if provider_id == "ollama" and not _server_can_call_ollama():
            continue
        try:
            reply = await call_provider(provider_id, messages)
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

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    prompt: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)
    provider: str = "auto"  # "auto" = try all until one works


def _request_messages(req: ChatRequest) -> list[dict]:
    if req.messages:
        return [{"role": m.role, "content": m.content} for m in req.messages]
    if req.prompt.strip():
        return [{"role": "user", "content": req.prompt.strip()}]
    raise HTTPException(status_code=400, detail="Send messages or a prompt.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/providers")
async def list_providers():
    return {"providers": get_available_providers()}


@app.post("/chat")
async def chat(req: ChatRequest):
    messages = _request_messages(req)
    if req.provider == "ollama" and not _server_can_call_ollama():
        raise HTTPException(
            status_code=503,
            detail="Ollama is local-only on Vercel. The page will try your machine's Ollama from the browser.",
        )
    if req.provider == "auto":
        reply, used_provider = await call_any_available(messages)
        return {"reply": reply, "provider": used_provider}
    else:
        reply = await call_provider(req.provider, messages)
        return {"reply": reply, "provider": PROVIDERS[req.provider]["label"]}


@app.get("/", response_class=HTMLResponse)
async def home():
    page = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLM Chat</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            html, body { height: 100%; }
            body {
                font-family: system-ui, -apple-system, sans-serif;
                background: #212121;
                color: #ececec;
            }
            .app { display: flex; flex-direction: column; height: 100%; max-width: 880px; margin: 0 auto; }
            header {
                display: flex; align-items: center; gap: 0.75rem;
                padding: 0.75rem 1rem; border-bottom: 1px solid #333;
                background: #171717;
            }
            header h1 { font-size: 1rem; font-weight: 600; white-space: nowrap; }
            select {
                flex: 1; min-width: 0; padding: 0.45rem 0.6rem; border-radius: 8px;
                border: 1px solid #333; background: #2f2f2f; color: #ececec; font-size: 0.9rem;
            }
            option:disabled { color: #777; }
            .new-chat {
                padding: 0.45rem 0.8rem; border: 1px solid #444; border-radius: 8px;
                background: transparent; color: #ececec; cursor: pointer; font-size: 0.85rem;
            }
            .new-chat:hover { background: #2f2f2f; }
            #thread {
                flex: 1; overflow-y: auto; padding: 1.25rem 1rem 1.5rem;
            }
            .empty {
                height: 100%; display: flex; flex-direction: column;
                align-items: center; justify-content: center; color: #888; text-align: center; gap: 0.5rem;
            }
            .empty h2 { color: #ececec; font-size: 1.5rem; font-weight: 600; }
            .row { display: flex; margin: 0.85rem 0; }
            .row.user { justify-content: flex-end; }
            .row.assistant { justify-content: flex-start; }
            .bubble {
                max-width: 80%; padding: 0.75rem 1rem; border-radius: 18px;
                white-space: pre-wrap; word-break: break-word; line-height: 1.5; font-size: 0.95rem;
            }
            .row.user .bubble { background: #2f2f2f; border-bottom-right-radius: 6px; }
            .row.assistant .bubble { background: #171717; border: 1px solid #2a2a2a; border-bottom-left-radius: 6px; }
            .row.error .bubble { background: #3b1d1d; border: 1px solid #7f1d1d; color: #fecaca; }
            .meta { font-size: 0.7rem; color: #777; margin-top: 0.35rem; }
            .typing { color: #888; }
            .composer-wrap { padding: 0.75rem 1rem 1.1rem; background: #212121; }
            .composer {
                display: flex; gap: 0.5rem; align-items: flex-end;
                background: #2f2f2f; border: 1px solid #3a3a3a; border-radius: 24px; padding: 0.5rem 0.55rem 0.5rem 1rem;
            }
            textarea {
                flex: 1; border: none; outline: none; resize: none; background: transparent;
                color: #ececec; font: inherit; max-height: 160px; min-height: 24px; line-height: 1.4;
            }
            #send {
                width: 36px; height: 36px; border: none; border-radius: 50%;
                background: #fff; color: #111; font-size: 1.1rem; cursor: pointer; flex-shrink: 0;
            }
            #send:disabled { opacity: 0.35; cursor: not-allowed; }
            .hint { text-align: center; font-size: 0.7rem; color: #666; margin-top: 0.45rem; }
        </style>
    </head>
    <body>
        <div class="app">
            <header>
                <h1>LLM Chat</h1>
                <select id="provider"><option value="auto">Auto (try all)</option></select>
                <button class="new-chat" id="newChat" type="button">New chat</button>
            </header>
            <div id="thread">
                <div class="empty" id="empty">
                    <h2>What can I help with?</h2>
                    <p>Ask a question, then keep chatting in this thread.</p>
                </div>
            </div>
            <div class="composer-wrap">
                <div class="composer">
                    <textarea id="prompt" rows="1" placeholder="Message..."></textarea>
                    <button id="send" type="button">↑</button>
                </div>
                <div class="hint">Enter to send · Shift+Enter for a new line</div>
            </div>
        </div>
        <script>
            const OLLAMA = {
                url: '__OLLAMA_CLIENT_URL__',
                model: '__OLLAMA_MODEL__',
                label: '__OLLAMA_LABEL__'
            };
            const messages = [];
            let busy = false;

            function escapeHtml(s) {
                return String(s)
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;');
            }

            function render() {
                const thread = document.getElementById('thread');
                if (!messages.length) {
                    thread.innerHTML = '<div class="empty" id="empty"><h2>What can I help with?</h2><p>Ask a question, then keep chatting in this thread.</p></div>';
                    return;
                }
                thread.innerHTML = messages.map(m => {
                    const cls = m.role === 'user' ? 'user' : (m.error ? 'assistant error' : 'assistant');
                    const meta = m.provider ? '<div class="meta">' + escapeHtml(m.provider) + '</div>' : '';
                    return '<div class="row ' + cls + '"><div class="bubble">' + escapeHtml(m.content) + meta + '</div></div>';
                }).join('');
                thread.scrollTop = thread.scrollHeight;
            }

            function payloadMessages() {
                return messages
                    .filter(m => !m.error && !m.pending && (m.role === 'user' || m.role === 'assistant'))
                    .map(m => ({ role: m.role, content: m.content }));
            }

            async function loadProviders() {
                const sel = document.getElementById('provider');
                try {
                    const res = await fetch('/providers');
                    const data = await res.json();
                    data.providers.forEach(p => {
                        const opt = document.createElement('option');
                        opt.value = p.id;
                        let label = p.label;
                        if (p.client_local) label += ' (this computer)';
                        else if (!p.available) label += ' (no key)';
                        opt.textContent = label;
                        opt.disabled = !p.available;
                        sel.appendChild(opt);
                    });
                } catch (e) {
                    console.error('Failed to load providers', e);
                }
            }

            async function callLocalOllama(history) {
                const res = await fetch(OLLAMA.url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: OLLAMA.model, messages: history })
                });
                const data = await res.json().catch(() => ({}));
                if (!res.ok) {
                    throw new Error(data.error || data.detail || ('Ollama HTTP ' + res.status));
                }
                const text = data.choices && data.choices[0] && data.choices[0].message
                    ? data.choices[0].message.content
                    : '';
                if (!text) throw new Error('Empty Ollama reply');
                return text;
            }

            async function callServer(history, provider) {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages: history, provider })
                });
                const data = await res.json().catch(() => ({}));
                return { ok: res.ok, data };
            }

            function autosize() {
                const el = document.getElementById('prompt');
                el.style.height = 'auto';
                el.style.height = Math.min(el.scrollHeight, 160) + 'px';
            }

            async function send() {
                const btn = document.getElementById('send');
                const input = document.getElementById('prompt');
                const text = input.value.trim();
                const provider = document.getElementById('provider').value;
                if (!text || busy) return;
                busy = true;
                btn.disabled = true;
                messages.push({ role: 'user', content: text });
                input.value = '';
                autosize();
                messages.push({ role: 'assistant', content: 'Thinking...', pending: true });
                render();
                const history = payloadMessages();
                try {
                    let reply, used;
                    if (provider === 'ollama') {
                        reply = await callLocalOllama(history);
                        used = OLLAMA.label + ' (local)';
                    } else {
                        const { ok, data } = await callServer(history, provider);
                        if (ok) {
                            reply = data.reply;
                            used = data.provider;
                        } else {
                            try {
                                reply = await callLocalOllama(history);
                                used = OLLAMA.label + ' (local fallback)';
                            } catch (localErr) {
                                throw new Error((data.detail || JSON.stringify(data)) + ' | Local Ollama: ' + localErr.message);
                            }
                        }
                    }
                    messages.pop();
                    messages.push({ role: 'assistant', content: reply, provider: used });
                } catch (e) {
                    messages.pop();
                    messages.push({ role: 'assistant', content: e.message, error: true, provider: 'Error' });
                } finally {
                    busy = false;
                    btn.disabled = false;
                    render();
                    input.focus();
                }
            }

            document.getElementById('send').addEventListener('click', send);
            document.getElementById('newChat').addEventListener('click', () => {
                if (busy) return;
                messages.splice(0, messages.length);
                render();
                document.getElementById('prompt').focus();
            });
            document.getElementById('prompt').addEventListener('input', autosize);
            document.getElementById('prompt').addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    send();
                }
            });
            loadProviders();
        </script>
    </body>
    </html>
    """
    return (
        page
        .replace("__OLLAMA_CLIENT_URL__", OLLAMA_CLIENT_URL)
        .replace("__OLLAMA_MODEL__", PROVIDERS["ollama"]["model"])
        .replace("__OLLAMA_LABEL__", PROVIDERS["ollama"]["label"])
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("example:app", host="127.0.0.1", port=8000, reload=True)
