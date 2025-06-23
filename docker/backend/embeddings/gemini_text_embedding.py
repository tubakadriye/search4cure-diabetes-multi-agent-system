# backnd/embeddings/gemini_text_embedding.py
import time
import tiktoken
import google.generativeai as genai

GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
MAX_TOKENS = 512      # Good default for embeddings
OVERLAP = 50          # Helps with context continuity

def chunk_text(text, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    """
    Split the text into overlapping chunks based on token count.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk = encoding.decode(chunk_tokens)
        chunks.append(chunk)
    return chunks

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_gemini_embedding(input_data, model=GEMINI_EMBEDDING_MODEL):
    """
    Use Gemini to embed input string or DataFrame row (with 'combined_info').
    """
    if isinstance(input_data, str):
        text = input_data
    else:
        text = input_data["combined_info"]
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []

    chunks = chunk_text(text)

    chunk_embeddings = []
    for chunk in chunks:
        time.sleep(1.5)
        chunk = chunk.replace("\n", " ")
        response = genai.embed_content(
            model=model,
            content=chunk,
            task_type="retrieval_document",
        )
        embedding = response["embedding"]
        chunk_embeddings.append(embedding)

    if isinstance(input_data, str):
        return chunk_embeddings[0]

    duplicated_rows = []
    for embedding in chunk_embeddings:
        new_row = input_data.copy()
        new_row["embedding"] = embedding
        duplicated_rows.append(new_row)
    return duplicated_rows
