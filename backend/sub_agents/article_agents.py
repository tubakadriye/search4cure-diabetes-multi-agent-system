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
MODEL = genai.GenerativeModel("gemini-2.0-flash")
AGENT_MODEL = "gemini-2.0-flash"
GCS_PROJECT = os.getenv("GCS_PROJECT")
GCS_BUCKET = os.getenv("GCS_BUCKET")

# Connect to the MongoDB collection
collection = mongodb_client[DB_NAME][COLLECTION_NAME]
db = mongodb_client[DB_NAME]

# Instantiate the GCS client and bucket
gcs_client = storage.Client(project=GCS_PROJECT)
gcs_bucket = gcs_client.bucket(GCS_BUCKET)


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
        text_summary = r.get("summary") or r.get("page_text", "")[:300]
        gcs_key = r.get("gcs_key", "")
        doc_id = r.get("_id")

        # Get Gemini citations/mentions from image
        try:
            cached = collection_ref.find_one({"_id": doc_id}, {"gemini_summary": 1})
            if cached and "gemini_summary" in cached:
                gemini = cached["gemini_summary"]
            else:
                page_bytes = get_image_from_gcs(gcs_bucket, gcs_key)
                page_image = Image.open(BytesIO(page_bytes))
                gemini = extract_info_from_page_image(page_image, title, page)
                collection_ref.update_one({"_id": doc_id}, {"$set": {"gemini_summary": gemini}})
        except Exception as e:
            gemini = f"Gemini image error: {e}"

        # Parse citations/figures only
        citations = extract_markdown_section(gemini, "Citations")
        figures = extract_markdown_section(gemini, "Figures/Tables")

        full_summary = f"### {title}, Page {page}\n**Summary**: {text_summary.strip()}\n\n**Citations**: {citations}\n\n**Figures/Tables**: {figures}"

        summaries.append(full_summary)

    return "\n\n".join(summaries)  # return top 5 summaries

import re

def extract_markdown_section(text: str, section_title: str) -> str:
    """
    Extracts a section from a markdown-formatted Gemini response by header name (e.g., "Citations").
    """
    pattern = rf"\*\*{re.escape(section_title)}:\*\*(.*?)(?=\n\*\*|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "Not found"



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
        response = MODEL.generate_content([prompt, image])
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

    collection_ref = db[COLLECTION_NAME]

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
        
    summaries = []
    for r in results[:5]:
        summary = gemini_page_image_summary(r, collection_ref)
    summaries.append(summary)
        
    return "\n\n".join(summaries)


article_page_vector_search_agent = Agent(
    name="article_page_vector_search_agent",
    model=AGENT_MODEL,
    description="Search academic papers (page-level) using a text query via multimodal embeddings.Returns the top 5 page-level summaries",
    instruction="You are an academic search agent doing text based research with sbert embeddings. Use the 'article_page_vector_search_tool' to find relevant page-level results from academic papers. Use sbert embedding.Do not use other embedding. Use collection docs_multimodal.",
    tools=[article_page_vector_search_tool],
)

vector_search_image_agent = Agent(
    name="vector_image_search_agent",
    model=AGENT_MODEL,
    description="Search PDFs via text or image query and summarize page images",
    instruction="You are a multimodal search agent. If the user provides an image or a descriptive text, use 'vector_search_image_tool' to return the top 5 academic page results with citations and summaries.Use clip_image as embedding if image is provided. Use clip embedding if the query is text. Do not use other embeddings. Use docs_multimodal collection.",
    tools=[vector_search_image_tool],
)

def gemini_page_image_summary(doc, collection_ref):
    doc_id = doc.get("_id")
    pdf_title = doc.get("pdf_title", "Unknown Title")
    page_number = doc.get("page_number", "?")
    gcs_key = doc.get("gcs_key", "")

    cached_doc = collection_ref.find_one({"_id": doc_id}, {"gemini_summary": 1})
    if cached_doc and cached_doc.get("gemini_summary"):
        summary = cached_doc["gemini_summary"]
    else:
        try:
            page_bytes = get_image_from_gcs(gcs_bucket, gcs_key)
            page_image = Image.open(BytesIO(page_bytes))
            summary = extract_info_from_page_image(page_image, pdf_title, page_number)
            collection_ref.update_one({"_id": doc_id}, {"$set": {"gemini_summary": summary}})
        except Exception as e:
            summary = f"Failed to fetch or process page image: {e}"

    citation = f"### {pdf_title}, Page {page_number}"
    return f"{citation}\n{summary.strip()}"



