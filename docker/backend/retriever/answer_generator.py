from io import BytesIO
from google import genai
from google.genai import types
from db import mongodb_client
from retriever.rag_retriever import vector_search
from dotenv import load_dotenv
import sys
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Gemini LLm to use
LLM = "gemini-2.0-flash"
# Names of the MongoDB database, collection and vector search index
DB_NAME = "diabetes_data"
COLLECTION_NAME = "docs_multimodal"
VS_INDEX_NAME = "multimodal_vector_index"
# Connect to the MongoDB collection

# Instantiate the Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def generate_answer(user_query: str, model: str) -> str:
    """
    Generate an answer to the user question using a Gemini multimodal LLM,
    integrating context from both text and images, with citation metadata.

    Args:
        user_query (str): User query string.
        model (str): Embedding model to use, either "voyage" or "clip".

    Returns:
        str: LLM generated answer.
    """
    collection = mongodb_client[DB_NAME][COLLECTION_NAME]
    # Run vector search using selected embedding space
    matched_docs = vector_search(user_query, model,collection, display_images=True)
    # Gather context text and images
    context_chunks = []
    images = []

    for doc in matched_docs:
        # Add text if available
        page_text = doc.get("page_text", "").strip()

        # Construct citation metadata
        pdf_title = doc.get("pdf_title", "Unknown Document")
        page_number = doc.get("page_number", "?")
        metadata = f"📄 **Source:** _{pdf_title}_ (Page {page_number})"

        if page_text:
            # Select up to 2 quotes/sentences as highlights
            sentences = [s.strip() for s in page_text.split(".") if s.strip()]
            highlights = ". ".join(sentences[:2]) + "." if sentences else page_text
            context_chunks.append(f"{metadata}\n> {highlights}")

        # Process image
        if "image_bytes" in doc:
            try:
                image = Image.open(BytesIO(doc["image_bytes"])).convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"Failed to load image from doc {doc.get('gcs_key', '')}: {e}")

    # Get the images from GCS and open them
    #images = [Image.open(BytesIO(get_image_from_gcs(key))) for key in gcs_keys]
    
    # Build the prompt
    context_text = "\n---\n".join(context_chunks)
    prompt = (
        "You are a researcher that uses both textual and visual information to answer questions.\n\n"
        "Use the following **textual context and images** to answer the user's question. "
        "Where appropriate, cite your answer using the source titles and page numbers.\n\n"
        f"### User Question:\n{user_query}\n\n"
        f"### Context:\n{context_text}"
    )
    #prompt = f"Answer the question based only on the provided context. If the context is empty, say I DON'T KNOW\n\nQuestion:{user_query}\n\nContext:\n"
    # Prompt to the LLM consisting of the system prompt and the images
    messages = [prompt] + images 
    # Get a response from the LLM
    response = gemini_client.models.generate_content(
        model=LLM,
        contents=messages,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    print(response)
    return response.text