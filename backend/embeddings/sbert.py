from typing import List

from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  #all-mpnet-base-v2

def get_sbert_embedding(text: str) -> List[float]:
    """
    Get SBERT embedding for text.

    Args:
        text (str): Input text.

    Returns:
        List[float]: Embedding vector as a list.
    """
    embedding = sbert_model.encode(text)  # usually returns a numpy array
    return embedding.tolist()