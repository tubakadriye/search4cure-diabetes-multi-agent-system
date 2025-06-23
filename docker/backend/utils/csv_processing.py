# backnd/utils/csv_processing.py

import datetime
import os
import pandas as pd
from tqdm import tqdm
from db.mongo_utils import insert_df_to_mongodb
from embeddings.gemini_text_embedding import get_gemini_embedding
from utils.attribute_combiner import combine_all_attributes
from utils.general_helpers import print_dataframe_info


def process_and_upload_csv(
    uploaded_csv, 
    datasets_col, 
    data_col, 
    embedding_fn=get_gemini_embedding
) -> int:    
    if not uploaded_csv:
        return 0
    #Check if file already uploaded
    file_name = uploaded_csv.name
    if datasets_col.find_one({"file_name": file_name}):
        print(f"⏭️ {file_name} already uploaded. Skipping.")
        return 0
    
    print(f"📂 Processing {file_name}...")

    # Initialize metadata tracking
    first_chunk = None
    total_rows = 0
    combined_missing = None

    # Read file into DataFrame
    try:
        if file_name.endswith(".csv"):
            chunk_iter = pd.read_csv(uploaded_csv, chunksize=500)
        elif file_name.endswith((".xlsx", ".xls")):
            chunk_iter = pd.read_excel(uploaded_csv, chunksize=500)
        elif file_name.endswith(".json"):
            chunk_iter = pd.read_json(uploaded_csv, lines=True, chunksize=500)
    except Exception as e:
        print(f"❌ Failed to read {file_name}: {e}")
        return 0

    dataset_id = None
    # === Insert Metadata ===
    for chunk_idx, chunk in enumerate(chunk_iter):
        if first_chunk is None:
            first_chunk = chunk.copy()
            combined_missing = chunk.isnull().sum()
            dataset_doc = {
                "file_name": file_name,
                "upload_date": datetime.now(),
                "n_columns": chunk.shape[1],
                "columns": chunk.columns.tolist(),
                "missing_values": chunk.isnull().sum().to_dict(),
                "file_type": os.path.splitext(file_name)[-1].replace(".", ""),
                "file_path": uploaded_csv,
                "column_types": chunk.dtypes.astype(str).to_dict(),
            }
            dataset_id = datasets_col.insert_one(dataset_doc).inserted_id
            print(f"✅ Inserted metadata for {file_name}")

        total_rows += len(chunk)
    
        # === Combine all attributes ===
        chunk = combine_all_attributes(chunk, exclude_columns=[])
        print(chunk[["combined_info"]].head(2))  # preview
    
        duplicated_data = []
        for row in tqdm(chunk.itertuples(index=False), total=len(chunk), desc="Embedding"):
            duplicated_rows = embedding_fn(row._asdict())
            duplicated_data.extend(duplicated_rows)

        if duplicated_data:
            df_dup = pd.DataFrame(duplicated_data)
            print(df_dup.head(2))
            try:
                total_inserted_tabular = insert_df_to_mongodb(df_dup, data_col, dataset_id)
                print(
                    f"📌 Chunk {chunk_idx + 1}: {len(df_dup)} documents inserted."
                )
            except Exception as e:
                print(f"❌ Error inserting chunk {chunk_idx + 1}: {e}")

    if dataset_id:
        datasets_col.update_one(
            {"_id": dataset_id},
            {"$set": {"n_rows": total_rows}}
        )
        print(f"✅ Finalized metadata for {file_name}: {total_rows} rows")
        return len(total_rows)
    
    print("⚠️ No embedded rows inserted.")
    return 0

    