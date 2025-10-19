# embedding_helper.py
import os, json, requests, math
from typing import List

API_BASE    = "https://psacodesprint2025.azure-api.net"
DEPLOYMENT  = "text-embedding-3-small"
API_VERSION = "2023-05-15"
SUB_KEY     = (os.getenv("PORTAL_SUB_KEY", "191897e716604e55b3f88ef227800633") or "").strip()

# must include subscription-key in query param
URL = (
    f"{API_BASE}/{DEPLOYMENT}/openai/deployments/{DEPLOYMENT}/embeddings"
    f"?api-version={API_VERSION}&subscription-key={SUB_KEY}"
)

HEADERS = {
    "Ocp-Apim-Subscription-Key": SUB_KEY,
    "Content-Type": "application/json",
    "Ocp-Apim-Trace": "true",
}

def get_embedding(text: str) -> List[float]:
    """
    Call Azure APIM 'text-embedding-3-small' and return the embedding vector.
    """
    if not text:
        return []
    payload = {
        "input": text,
        "user": "excel_scan",
        "input_type": "query",
        "model": DEPLOYMENT,
    }

    try:
        r = requests.post(URL, headers=HEADERS, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["data"][0]["embedding"]
    except requests.HTTPError:
        try:
            detail = r.json()
        except Exception:
            detail = {"raw": r.text}
        print(f"❌ [HTTP {r.status_code}] {json.dumps(detail, indent=2)}")
        return []
    except Exception as e:
        print(f"❌ [embedding error] {e}")
        return []

def get_cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

if __name__ == "__main__":
    v = get_embedding("hello world")
    print("dims:", len(v), "first5:", v[:5])
