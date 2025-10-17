import os, json, requests

API_BASE   = "https://psacodesprint2025.azure-api.net/gpt-5-mini/openai"
DEPLOYMENT = "gpt-5-mini"
API_VERSION = "2025-01-01-preview"
SUB_KEY = (os.getenv("PORTAL_SUB_KEY", "0aca49599682440084da0594a81adb80") or "").strip()

URL = f"{API_BASE}/deployments/{DEPLOYMENT}/chat/completions?api-version={API_VERSION}&subscription-key={SUB_KEY}"

HEADERS = {
    "Ocp-Apim-Subscription-Key": SUB_KEY,
    "Content-Type": "application/json",
    "Ocp-Apim-Trace": "true",
}

def ask_gpt5(
    user_message: str,
    system_prompt: str = "You are a concise assistant. Always answer with plain text.",
    max_completion_tokens: int = 100000
) -> str:
    """
    Send a message to gpt-5-mini via Azure APIM and return assistant text.
    Returns readable error details if the server responds with 4xx/5xx.
    """

    payload = {
        "model": DEPLOYMENT,  # some APIM policies require model explicitly
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_completion_tokens": int(max_completion_tokens),
        "response_format": {"type": "text"},
    }

    try:
        r = requests.post(URL, headers=HEADERS, json=payload, timeout=120)
        if not r.ok:
            # Surface server error details to help debug 400s
            try:
                err = r.json()
            except Exception:
                err = {"raw": r.text}
            return f"[HTTP {r.status_code}] {json.dumps(err, indent=2)}"
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.RequestException as e:
        return f"[Network error] {e}"
    except Exception as e:
        return f"[Unexpected error] {e}"

if __name__ == "__main__":
    print(ask_gpt5("Give me the bio of Tzuyu from TWICE."))
