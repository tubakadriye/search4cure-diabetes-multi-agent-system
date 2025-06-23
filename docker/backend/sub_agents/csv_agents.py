

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from db.mongodb_client import MONGODB_URI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever
import os
from dotenv import load_dotenv
from google.adk.agents import Agent
# @title Import necessary libraries
import os
import asyncio
from google.adk.agents import Agent
from google.adk.agents import LlmAgent
#from google.adk.models.lite_llm import LiteLlm # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
 # For creating message Content/Parts
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime

import logging
logging.basicConfig(level=logging.ERROR)

load_dotenv()

DB_NAME = os.getenv("MONGO_DB")
ATLAS_VECTOR_SEARCH_INDEX = "csv_vector_index"
CSV_COLLECTION= "records_embeddings"
GEMINI_EMBEDDING_MODEL = "models/embedding-001"#"models/gemini-embedding-exp-03-07"
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"
AGENT_MODEL = MODEL_GEMINI_2_0_FLASH 


embedding_model = GoogleGenerativeAIEmbeddings(
    model=GEMINI_EMBEDDING_MODEL,
    task_type="RETRIEVAL_DOCUMENT"
)

vector_store_csv_files = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGODB_URI,
    namespace=DB_NAME + "." + CSV_COLLECTION,
    embedding=embedding_model,
    index_name=ATLAS_VECTOR_SEARCH_INDEX,
    text_key="combined_info",
)

def csv_files_vector_search_tool(query: str, k: int = 5):
    """
    Perform a vector similarity search on safety procedures.

    Args:
        query (str): The search query string.
        k (int, optional): Number of top results to return. Defaults to 5.

    Returns:
        list: List of tuples (Document, score), where Document is a record
              and score is the similarity score (lower is more similar).

    Note:
        Uses the global vector_store_csv_files for the search.
    """

    vector_search_results = vector_store_csv_files.similarity_search_with_score(
        query=query, k=k
    )
    return vector_search_results



# Create the agent
csv_files_vector_search_agent = Agent(
    name="csv_files_vector_search_agent",
    model=AGENT_MODEL,
    description="Performs a vector similarity search on safety procedures from CSV files.",
    instruction="You are an agent that performs a semantic similarity search on patient data about diabetes"
                "stored in CSV files using the 'csv_files_vector_search_tool'. "
                "Return the most relevant entries from the database based on the user's query.",
    tools=[csv_files_vector_search_tool],
)

print(f"Agent '{csv_files_vector_search_agent.name}' created using model '{AGENT_MODEL}'.")


def hybrid_search_tool(query: str):
    """
    Perform a hybrid (vector + full-text) search on safety procedures.

    Args:
        query (str): The search query string.

    Returns:
        list: Relevant safety procedure documents from hybrid search.

    Note:
        Uses both vector_store_csv_files and record_text_search_index.
    """

    hybrid_search = MongoDBAtlasHybridSearchRetriever(
        vectorstore=vector_store_csv_files,
        search_index_name="record_text_search_index",
        top_k=5,
    )

    hybrid_search_result = hybrid_search.get_relevant_documents(query)

    return hybrid_search_result




## Generalized Record Document Creator
class GenericRecord(BaseModel):
    # Flexible for any CSV columns
    data: Dict[str, Any]
    combined_info: Optional[str] = None
    embedding: Optional[List[float]] = None
    dataset_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True


from typing import List

from pydantic import BaseModel, Field


## Function to Create the Document
def create_generic_record_document(row: Dict[str, Any], dataset_id: str) -> dict:
    """
    Create a standardized document for any CSV record with flexible columns.

    Args:
        row (Dict[str, Any]): The actual CSV row as a dictionary.
        dataset_id (str): Reference to the parent dataset document.

    Returns:
        dict: Cleaned and standardized MongoDB document.
    """
    try:
        # Clean keys: remove leading/trailing whitespace, lowercase, etc. (optional)
        cleaned_row = {k.strip(): v for k, v in row.items()}

        # Create combined_info string for text search
        combined_info = " ".join(f"{k}: {v}" for k, v in cleaned_row.items())

        # Package into a generic Pydantic record
        record = GenericRecord(
            data=cleaned_row,
            combined_info=combined_info,
            dataset_id=dataset_id
        )

        return record.dict()
    except Exception as e:
        raise ValueError(f"Invalid record data: {e!s}")


# Tool to add new record
def create_new_record(new_record: Dict[str, any]) -> dict:
    """
    Create and validate a new generic CSV record.

    Args:
        new_record (dict): Dictionary containing a row from a CSV file. 
                           Must include 'dataset_id' as a key.

    Returns:
        dict: Validated and formatted record document.

    Raises:
        ValueError: If the input data is invalid or incomplete.

    Note:
        Uses Pydantic for data validation via create_generic_record_document function.
    """
    try:
        dataset_id = new_record.pop("dataset_id", None)
        if not dataset_id:
            raise ValueError("Missing required field 'dataset_id' in the new record.")

        document = create_generic_record_document(new_record, dataset_id)
        return document

    except Exception as e:
        raise ValueError(f"Error creating new record: {e}")
    


# Create the agent
csv_files_vector_search_agent = Agent(
    name="csv_files_vector_search_agent",
    model=AGENT_MODEL,
    description="Performs a vector similarity search on safety procedures from CSV files.",
    instruction="You are an agent that performs a semantic similarity search on patient data about diabetes"
                "stored in CSV files using the 'csv_files_vector_search_tool'. "
                "Return the most relevant entries from the database based on the user's query.",
    tools=[csv_files_vector_search_tool],
)

print(f"Agent '{csv_files_vector_search_agent.name}' created using model '{AGENT_MODEL}'.")


hybrid_search_agent = Agent(
    name="hybrid_search_agent",
    model=AGENT_MODEL,
    description="Performs a hybrid (vector + full-text) search on CSV safety records.",
    instruction="You are an agent that retrieves the most relevant CSV records related to safety using both full-text and vector similarity search. Use the 'hybrid_search_tool'.",
    tools=[hybrid_search_tool],
)

create_record_agent = Agent(
    name="create_record_agent",
    model=AGENT_MODEL,
    description="Creates and validates new records for a CSV dataset.",
    instruction="You are a data creation agent. When a user submits a new CSV row with a dataset ID, validate and format it using the 'create_new_record' tool.",
    tools=[create_new_record],
)
