from io import BytesIO
from embeddings.clip import get_clip_embedding
from embeddings.gemini_text_embedding import get_gemini_embedding
from embeddings.sbert import get_sbert_embedding
#from embeddings.voyage import get_voyage_embedding
from utils.gcs_utils import get_image_from_gcs
from typing import Union
from typing import List, Union
from PIL import Image
from db.mongodb_client import mongodb_client
import os
from google.cloud import storage
import google.generativeai as genai
from typing import Union, List, Dict, Any

# Google AI SDK setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
LLM = genai.GenerativeModel("gemini-2.0-flash")

# Constants
DB_NAME = "diabetes_data"
COLLECTION_NAME = "docs_multimodal"
VS_INDEX_NAME = "multimodal_vector_index"
# Set GCS project and bucket
GCS_PROJECT = os.getenv("GCS_PROJECT")
GCS_BUCKET = os.getenv("GCS_BUCKET")

# Initialize your GCS client and bucket
gcs_client = storage.Client(project=GCS_PROJECT)
gcs_bucket = gcs_client.bucket(GCS_BUCKET)

# Connect to the MongoDB collection
collection = mongodb_client[DB_NAME][COLLECTION_NAME]

from langchain_google_genai import ChatGoogleGenerativeAI

# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
#                             temperature=0,
#                             max_tokens=None,
#                             timeout=None,
#                             max_retries=3,
#                             # other params...
# )


def vector_search(
    user_query: Union[str, Image.Image],
    model: str,
    collection,
    display_images: bool = True
) -> List[Dict[str, Any]]:
    """
    Perform vector search using CLIP or SBERT, return matched GCS keys.
    
    Args:
        user_query (str or Image): Text or image input.
        model (str): One of "voyage", "clip", or "sbert".
        collection(str): MongoDB collection.
        display_images (bool): Show images if True.

    Returns:
        List[str]: GCS keys of matched items.
    """
    # Map models to stored embedding fields
    MODEL_PATH_MAP = {
        "sbert": "sbert_text_embedding",
        "clip": "clip_text_embedding",  # for text query
        "clip_image": "clip_image_embedding",  # if image input
        "voyage": "voyage_embedding"  # optional
    }

    embedding_path = MODEL_PATH_MAP.get(model)

   
    if embedding_path is None:
        raise ValueError(f"Unsupported or unconfigured model: {model}")
    
    # Compute query embedding
    if model.startswith("clip"):
        emb = get_clip_embedding(user_query) if isinstance(user_query, Image.Image) or model=="clip" else None
    else:
        emb = get_sbert_embedding(user_query)


    if emb is None:
        raise ValueError("Failed to compute embedding")

    # Ensure query_embedding is a list of floats
    query_embedding = emb.tolist() if not isinstance(emb, list) else emb
    
    # Run MongoDB vectorSearch pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": VS_INDEX_NAME,
                "queryVector": query_embedding,
                "path": embedding_path,
                "numCandidates": 150,
                "limit": 20,
            }
        },
        {
            "$project": {
                "_id": 0,
                "gcs_key": 1,
                "width": 1,
                "height": 1,
                "score": {"$meta": "vectorSearchScore"},
                "pdf_title" : 1,
                "page_number":1,
                "sbert_text_embedding": 1,
                "clip_text_embedding": 1,
                "clip_image_embedding":1,
                "page_text": 1,
                "url":1
            }
        },
    ]

    results = collection.aggregate(pipeline)
    matched_docs = []

    for result in results:
        gcs_key = result["gcs_key"]
        result["image_bytes"] = get_image_from_gcs(gcs_bucket,gcs_key) 
        insights = _summarize_with_gemini(result["page_text"])
        result.update(insights)
        if display_images:
            img = Image.open(BytesIO(result["image_bytes"]))
            print(f"{result['score']}\n")
            img.show()
        matched_docs.append(result)

    return matched_docs


def vector_search_for_csv(user_query, collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the user query
    query_embedding = get_gemini_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Define the vector search pipeline
    vector_search_stage = {
        "$vectorSearch": {
            "index": "vector_index_with_filter",
            "queryVector": query_embedding,
            "path": "embedding",
            "numCandidates": 150,  # Number of candidate matches to consider
            "limit": 5,  # Return top 4 matches
        }
    }

    unset_stage = {
        "$unset": "embedding"  # Exclude the 'embedding' field from the results
    }

    project_stage = {
        "$project": {
            "_id": 0,  # Exclude the _id field,
            "combined_info": 1,
            "score": {
                "$meta": "vectorSearchScore"  # Include the search score
            },
        }
    }

    pipeline = [vector_search_stage, unset_stage, project_stage]

    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)

def summarize_and_link(page_text: str) -> dict:
    system_prompt = """
    You are a research assistant. Given a page of text from a research paper, summarize its content,
    extract all names of treatments, methods, or referenced articles mentioned, and provide any clues 
    about further research directions (e.g., related studies or innovations).
    
    Respond in this format:
    {
      "summary": "...",
      "mentions": [...],
      "linked_articles": [{"title": "...", "url": "..."}]
    }
    """
#     messages = [
#     (
#         "system",
#         system_prompt,
#     ),
#     (f"\n\nPage Content:\n{page_text}"),
# ]
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Page Content:\n{page_text}"}
    ]
    response = llm.invoke(messages)
    return parse_json_response(response.content)

import json

def _summarize_with_gemini(page_text: str) -> Dict[str, Any]:
    prompt = (
        "You are an academic research assistant. "
        "Summarize this page and extract treatments, methods, linked articles.\n\n"
        + page_text
    )
    resp = LLM.generate_content([prompt])
    try:
        data = genai_response_to_json(resp.text)
    except:
        data = {"summary": resp.text, "mentions": [], "linked_articles": []}
    return data

def genai_response_to_json(text: str) -> Dict[str, Any]:
    import json
    return json.loads(text)


def parse_json_response(response: str) -> dict:
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"summary": response, "mentions": [], "linked_articles": []}




