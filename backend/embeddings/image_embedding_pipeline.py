# embeddings/image_embedding_pipeline.py

from PIL import Image
from io import BytesIO
from tqdm import tqdm

def embed_docs_with_clip(docs, get_clip_embedding):
    """
    Embeds a list of image-docs using CLIP. Expects 'image' in each doc.
    Adds 'clip_embedding' key.
    """
    embedded_docs = []
    for doc in tqdm(docs, desc="Embedding docs with CLIP"):
        img = Image.open(BytesIO(doc["image"]))
        doc["clip_embedding"] = get_clip_embedding(img)
        del doc["image"]  # Remove image to reduce memory/size
        embedded_docs.append(doc)
    return embedded_docs

def embed_docs_with_voyage(docs, get_voyage_embedding):
    """
    Embeds a list of image-docs using Voyage. Expects 'image' in each doc.
    Adds 'voyage_embedding' key.
    """
    embedded_docs = []
    for doc in tqdm(docs, desc="Embedding docs with Voyage"):
        img = Image.open(BytesIO(doc["image"]))
        doc["voyage_embedding"] = get_voyage_embedding(img, task="document")
        del doc["image"]
        embedded_docs.append(doc)
    return embedded_docs
