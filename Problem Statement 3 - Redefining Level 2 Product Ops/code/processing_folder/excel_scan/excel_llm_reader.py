import pandas as pd
from typing import List, Tuple
from embedding_helper import get_embedding, get_cosine_similarity

def semantic_excel_search(query: str, 
                          xlsx_path: str = "/Users/zian/Documents/PSA Hackathon/PSA_Mediocre/Problem Statement 3 - Redefining Level 2 Product Ops/Info/Case Log.xlsx",
                          threshold: float = 0.7) -> List[Tuple[str, float]]:
    """
    Search an Excel file for semantically similar text to a given query.
    
    Args:
        query (str): The text you want to check for.
        xlsx_path (str): Path to the Excel file.
        threshold (float): Minimum cosine similarity (0–1) for a match.

    Returns:
        List of (matched_text, similarity_score) tuples.
    """
    #df = pd.read_excel(xlsx_path, header=None)
    df = pd.read_excel(xlsx_path, header=None).head(5)

    query_emb = get_embedding(query)
    if not query_emb:
        print("❌ Could not get query embedding.")
        return []

    matches = []
    for _, row in df.iterrows():
        for cell in row:
            if isinstance(cell, str) and cell.strip():
                cell_emb = get_embedding(cell)
                sim = get_cosine_similarity(query_emb, cell_emb)
                if sim >= threshold:
                    matches.append((cell, sim))

    matches.sort(key=lambda x: x[1], reverse=True)
    print(f"✅ Found {len(matches)} matches with similarity ≥ {threshold}")
    return matches


if __name__ == "__main__":
    query = "Notification: SMS TCK-936729 — Detected an ANSI X12 301 data mismatch for vessel MV SILVER CURRENT/43C. The COARRI message indicates discharge finished at bay 22, b. Kindly verify urgently."
    results = semantic_excel_search(query)
    for text, score in results[:5]:
        print(f"{score:.3f} → {text}")
