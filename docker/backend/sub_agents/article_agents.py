from dotenv import load_dotenv
import os
from retriever.rag_retriever import vector_search
from db.mongodb_client import mongodb_client
import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO
from google.cloud import storage
#from pymongo import MongoClient
from PIL import Image
from io import BytesIO
from typing import Dict, Any, Optional, List
from google.adk.agents import Agent

DB_NAME = "diabetes_data"
COLLECTION_NAME = "docs_multimodal"
VS_INDEX_NAME = "multimodal_vector_index"
# Connect to the MongoDB collection
collection = mongodb_client[DB_NAME][COLLECTION_NAME]
db = mongodb_client[DB_NAME]
GCS_PROJECT = os.getenv("GCS_PROJECT")
GCS_BUCKET = os.getenv("GCS_BUCKET")
# Instantiate the GCS client and bucket
gcs_client = storage.Client(project=GCS_PROJECT)
gcs_bucket = gcs_client.bucket(GCS_BUCKET)
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"
AGENT_MODEL = MODEL_GEMINI_2_0_FLASH 

def article_page_vector_search_tool(
    query: str,
    model: str = "sbert",
    collection_name: str = COLLECTION_NAME,
    )-> str:
    """
    Search academic papers (page-level) using a text query via multimodal embeddings.
    Returns the top 5 page-level summaries with citations.

    Args:
        query (str): The textual search query.
        model (str): Embedding model ("sbert", "clip", "voyage").
        collection (str): MongoDB collection name to search in.

    Returns:
        str: Top 5 matching pages, each with citation and summary.
        
    """
    collection_ref = mongodb_client[DB_NAME][collection_name]
    results = vector_search(query, model=model, collection=collection_ref, display_images=False)
    if not results:
        return "No relevant pages found."
    
    summaries = []
    for r in results[:5]:
        title = r.get("pdf_title", "Unknown Title")
        page = r.get("page_number", "?")
        text = r.get("summary") or r.get("page_text", "")[:300]
        summaries.append(f"[{title}, Page {page}]: {text.strip()}")

    return "\n\n".join(summaries)  # return top 5 summaries




model = genai.GenerativeModel("gemini-2.0-flash") #gemini-pro-vision

def extract_info_from_page_image(image: Image.Image, pdf_title: str, page_number: int) -> str:
    """
    Use Gemini to extract citations, figures/tables, and a summary from a page image.
    """
    #response = requests.get(gcs_url)
    #image = Image.open(BytesIO(response.content))

    prompt = f"""
    You are analyzing a scientific paper page image from the PDF titled: "{pdf_title}", page {page_number}.
    Please extract the following:
    1. All citations on the page (e.g., [1], Smith et al. 2020).
    2. Any tables or figures, including titles or captions.
    3. A brief summary of the page content.
    
    Return output in markdown with the following format:
    **Citations:** ...
    **Figures/Tables:** ...
    **Summary:** ...
    """

    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Gemini processing error: {e}"
    


from io import BytesIO
from google.cloud import storage

def get_image_from_gcs(gcs_bucket, key: str) -> bytes:
    """
    Download image bytes from GCS.

    Args:
        gcs_bucket: GCS bucket instance.
        key (str): Blob key in the bucket.

    Returns:
        bytes: Image bytes.
    """
    blob = gcs_bucket.blob(key)
    return blob.download_as_bytes()




def vector_search_image_tool(
    collection_name: str,
    image_bytes: Optional[bytes] = None,
    text_query: Optional[str] = None,  
    ) -> str:
    """
    Search academic papers using either an image or text query. If text is provided, it generates
    a CLIP embedding from the query and searches against page images. If image is provided, it
    searches using image embeddings.

    Args:
        image_bytes (bytes, optional): Query image content.
        text_query (str, optional): Text query to generate image embedding.
        collection_name (str): MongoDB collection name.

    Returns:
        str: Top 5 matching pages with citation and Gemini summary.
        
    """

    collection_ref = db[collection_name]

    # Determine search mode
    if image_bytes:
        try:
            query_image = Image.open(BytesIO(image_bytes))
        except Exception as e:
            return f"Invalid image input: {e}"
        results = vector_search(query_image, model="clip_image", collection=collection_ref, display_images=False)

    elif text_query:
        results = vector_search(text_query, model="clip", collection=collection_ref, display_images=False)

    else:
        return "Either image_bytes or text_query must be provided."
    
    if not results:
        return "No matching results found."
        
    output = []
    for r in results[:5]:
        pdf_title = r.get("pdf_title", "Unknown Title")
        page_number = r.get("page_number", -1)
        gcs_key = r.get("gcs_key", "")
        doc_id = r.get("_id")  # Ensure this is present in your search result

        # Check for existing gemini_summary in DB
        cached_doc = collection_ref.find_one({"_id": doc_id}, {"gemini_summary": 1})
        if cached_doc and cached_doc.get("gemini_summary"):
            summary = cached_doc["gemini_summary"]
        else:
            # If not cached, fetch image + generate Gemini summary
            try:
                page_bytes = get_image_from_gcs(gcs_bucket, gcs_key)
                page_image = Image.open(BytesIO(page_bytes))
            except Exception as e:
                return f"Failed to fetch page image: {e}"
            else:
                summary = extract_info_from_page_image(page_image, pdf_title, page_number)

                # Save summary to DB
                collection_ref.update_one(
                    {"_id": doc_id},
                    {"$set": {"gemini_summary": summary}},
                )

        citation = f"### {pdf_title}, Page {page_number}"
        output.append(f"{citation}\n{summary.strip()}")
        
    return "\n\n".join(output)


article_page_search_agent = Agent(
    name="article_page_search_agent",
    model=AGENT_MODEL,
    description="Search academic papers (page-level) using a text query via multimodal embeddings.Returns the top 5 page-level summaries with citations.",
    instruction="You are an academic search agent. Use the 'article_page_vector_search_tool' to find relevant page-level results from academic papers. ",
    tools=[article_page_vector_search_tool],
)

vector_search_image_agent = Agent(
    name="vector_search_image_agent",
    model=AGENT_MODEL,
    description="Search academic papers using either an image or text query. If text is provided, it generates a CLIP embedding from the query and searches against page images. If image is provided, it searches using image embeddings.",
    instruction="You are a multimodal search agent. If the user provides an image or a descriptive text, use 'vector_search_image_tool' to return the top 5 academic page results with citations and summaries.",
    tools=[vector_search_image_tool],
)

