import pandas as pd
from pymongo.errors import BulkWriteError


def insert_df_to_mongodb(df, collection,dataset_id,  batch_size=1000):
    """
    Insert a pandas DataFrame into a MongoDB collection.

    Parameters:
    df (pandas.DataFrame): The DataFrame to insert
    collection (pymongo.collection.Collection): The MongoDB collection to insert into
    dataset_id: Id of the file
    batch_size (int): Number of documents to insert in each batch

    Returns:
    int: Number of documents successfully inserted
    """
    total_inserted = 0

    # Convert DataFrame to list of dictionaries
    records = df.to_dict("records")

    for record in records:
        record["dataset_id"] = dataset_id

    # Insert in batches
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        try:
            result = collection.insert_many(batch, ordered=False)
            total_inserted += len(result.inserted_ids)
            print(
                f"Inserted batch {i//batch_size + 1}: {len(result.inserted_ids)} documents"
            )
        except BulkWriteError as bwe:
            total_inserted += bwe.details["nInserted"]
            print(
                f"Batch {i//batch_size + 1} partially inserted. {bwe.details['nInserted']} inserted, {len(bwe.details['writeErrors'])} failed."
            )

    return total_inserted