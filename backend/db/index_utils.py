from pymongo.operations import SearchIndexModel

def create_vector_index(db, collection_name, index_name, field_name, num_dimensions):
    index_model = {
        "name": index_name,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": field_name,
                    "numDimensions": num_dimensions,
                    "similarity": "cosine"
                }
            ]
        }
    }

    try:
        db.command("createSearchIndex", collection_name, **index_model)
        print(f"✅ Created index '{index_name}' on '{collection_name}'")
    except Exception as e:
        print(f"❌ Error creating index: {e}")



def create_multivector_index(db, collection_name, index_name):
    """
    Creates a multi-vector search index on the given MongoDB collection.
    Only runs if the index doesn't already exist.
    """
    collection = db[collection_name]

    existing_indexes = collection.index_information()
    existing_search_indexes = collection.list_search_indexes()

    if any(index["name"] == index_name for index in existing_search_indexes):
        print(f"Index '{index_name}' already exists. Skipping creation.")
        return

    index_definition = {
        "name": index_name,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "sbert_text_embedding",
                    "numDimensions": 384,
                    "similarity": "cosine",
                },
                {
                    "type": "vector",
                    "path": "clip_text_embedding",
                    "numDimensions": 512,
                    "similarity": "cosine",
                },
                {
                    "type": "vector",
                    "path": "clip_image_embedding",
                    "numDimensions": 512,
                    "similarity": "cosine",
                },
            ]
        },
    }

    collection.create_search_index(model=index_definition)
    print(f"Created index: {index_name}")


# Programmatically create vector search index for both colelctions


def setup_vector_search_index_with_filter(
    collection, index_definition, index_name="vector_index_with_filter"
):
    """
    Setup a vector search index for a MongoDB collection.

    Args:
    collection: MongoDB collection object
    index_definition: Dictionary containing the index definition
    index_name: Name of the index (default: "vector_index_with_filter")
    """
    new_vector_search_index_model = SearchIndexModel(
        definition=index_definition,
        name=index_name,
    )

    # Create the new index
    try:
        result = collection.create_search_index(model=new_vector_search_index_model)
        print(f"Creating index '{index_name}'...")
        # time.sleep(20)  # Sleep for 20 seconds
        print(f"New index '{index_name}' created successfully:", result)
    except Exception as e:
        print(f"Error creating new vector search index '{index_name}': {e!s}")
