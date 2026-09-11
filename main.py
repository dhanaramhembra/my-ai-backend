print(">>> STARTING SMART ROUTER...")
import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

# ============================================
# MULTI-KEY GROQ + FALLBACK PROVIDERS
# ============================================
PROVIDERS = []

# 5 Groq Keys rotate karo
for i in range(1, 6):
    key = os.environ.get(f"GROQ_API_KEY_{i}")
    if key:
        PROVIDERS.append({
            "name": f"Groq-{i}",
            "api_key": key,
            "base_url": "https://api.groq.com/openai/v1",
            "model": "openai/gpt-oss-120b",
        })

# Gemini (fallback)
if os.environ.get("GEMINI_API_KEY"):
    PROVIDERS.append({
        "name": "Gemini",
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-1.5-flash",
    })

# Together AI (fallback)
if os.environ.get("TOGETHER_API_KEY"):
    PROVIDERS.append({
        "name": "Together",
        "api_key": os.environ.get("TOGETHER_API_KEY"),
        "base_url": "https://api.together.xyz/v1",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    })

print(f">>> Total providers loaded: {len(PROVIDERS)}")

@app.post("/chat")
async def chat(req: ChatRequest):
    last_error = None

    for provider in PROVIDERS:
        if not provider["api_key"]:
            continue

        try:
            client = OpenAI(
                api_key=provider["api_key"],
                base_url=provider["base_url"],
            )

            response = client.chat.completions.create(
                model=provider["model"],
                messages=[{"role": "user", "content": req.message}],
                stream=True,
            )

            print(f">>> Using: {provider['name']}")

            def stream():
                try:
                    for chunk in response:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield delta
                except Exception as e:
                    print(f">>> Stream error {provider['name']}: {e}")

            return StreamingResponse(stream(), media_type="text/plain")

        except Exception as e:
            print(f">>> {provider['name']} failed: {str(e)[:100]}")
            last_error = str(e)
            continue

    return {"error": f"All failed. Last: {last_error}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
