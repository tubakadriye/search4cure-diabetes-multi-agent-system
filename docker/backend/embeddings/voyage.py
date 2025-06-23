
from typing import List

from PIL import Image
from voyageai import Client

# Instantiate the Voyage AI client
voyageai_client = Client()

def get_voyage_embedding(data: Image.Image | str, input_type: str) -> List:
    """
    Get Voyage AI embeddings for images and text.

    Args:
        data (Image.Image | str): An image or text to embed.
        input_type (str): Input type, either "document" or "query".

    Returns:
        List: Embeddings as a list.
    """
    embedding = voyageai_client.multimodal_embed(
        inputs=[[data]], model="voyage-multimodal-3", input_type=input_type
    ).embeddings[0]
    return embedding