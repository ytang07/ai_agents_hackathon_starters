import datasets
from config import (HF_DATASET, MAX_DATASET_LENGTH, INGESTION_BATCH_SIZE, 
                    LANCEDB_TABLE_NAME, DATASET_SPLIT, create_or_open_lancedb_table)
import base64
from inference_utils import get_embedding
import concurrent.futures

def is_primitive(value):
    """Helper function to check if a value is of a primitive type."""
    return isinstance(value, (int, float, str, bool, bytes))

def yield_dataset_batches():
    """
    Yield dataset batches as python lists of dictionaries, containing only primitive types.
    """
    dataset = datasets.load_dataset(HF_DATASET, split=DATASET_SPLIT)

    for i in range(0, MAX_DATASET_LENGTH, INGESTION_BATCH_SIZE):
        batch = dataset.select(range(i, min(i + INGESTION_BATCH_SIZE, MAX_DATASET_LENGTH)))
        pandas_batch = batch.to_pandas().to_dict(orient='records')
        
        cleaned_batch_for_ingestion = []
        for record in pandas_batch:
            cleaned_record = {}
            for key, value in record.items():
                if key == "image":
                    if isinstance(value, dict) and "bytes" in value:
                        cleaned_record["image"] = value["bytes"]
                    elif isinstance(value, bytes): # If it's already bytes
                        cleaned_record["image"] = value
                    else:
                        cleaned_record["image"] = b'' 
                elif is_primitive(value):
                    cleaned_record[key] = value
            cleaned_batch_for_ingestion.append(cleaned_record)
        yield cleaned_batch_for_ingestion

def ingest_batch_lancedb(batch):
    """
    Ingest data to LanceDB.
    """
    table = create_or_open_lancedb_table()

    image_api_input = []
    for record in batch:
        img_bytes_val = record.get('image', b'')
        if img_bytes_val:
            image_api_input.append(f"data:image/jpeg;base64,{base64.b64encode(img_bytes_val).decode('utf-8')}")
        else:
            image_api_input.append(None)

    embeddings = list(range(512)) * len(batch)
    future_to_index = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for index, img_data_uri in enumerate(image_api_input):
            if img_data_uri:
                future = executor.submit(get_embedding, img_data_uri)
                future_to_index[future] = index
            else: 
                embeddings[index] = [0.0]*512


        for future in concurrent.futures.as_completed(future_to_index):
            original_index = future_to_index[future]
            try:
                embeddings[original_index] = future.result()
            except Exception as exc:
                print(f'Generating embedding for index {original_index} generated an exception: {exc}')
                embeddings[original_index] = [0.0]*512

    data_to_add = []
    for i, record in enumerate(batch):
        
        record['vector'] = embeddings[i]
        
        record['image'] = record.get('image', b'') 

        record.pop('Unnamed: 0', None)
        data_to_add.append(record)
        
    if data_to_add:
        table.add(data_to_add)
        print(f"Ingested batch of size {len(data_to_add)}")
    else:
        print("Skipped empty batch for ingestion.")

def ingest_parallel():
    """
    Ingest data in parallel using ThreadPoolExecutor and wait for completion.
    """
    futures = [] 
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for batch in yield_dataset_batches():
            future = executor.submit(ingest_batch_lancedb, batch)
            futures.append(future)

        print(f"Submitted {len(futures)} batches for ingestion...")
        completed_count = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
                completed_count += 1
                print(f"Completed ingestion task {completed_count}/{len(futures)}")
            except Exception as exc:
                print(f'An ingestion task generated an exception: {exc}')
    print("All ingestion tasks finished.")


if __name__ == "__main__":
    print("Creating/Opening LanceDB table...")
    table = create_or_open_lancedb_table()
    if table is not None:
        print(f"Using table: {LANCEDB_TABLE_NAME}")
        print("Starting parallel ingestion...")
        ingest_parallel() 
        print("Ingestion finished.")
        print(f"Table schema: {table.schema}")
    else:
        print("Failed to create or open LanceDB table.")


