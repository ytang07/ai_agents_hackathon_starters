import os
import lancedb
from lancedb.pydantic import Vector, LanceModel
import datasets
from pydantic import create_model

LANCEDB_URI = "db://wikipedia-test-9cusod"
LANCEDB_API_KEY = os.environ.get("LANCEDB_API_KEY")
LANCEDB_TABLE_NAME =  "gmi-demo-cub"  #"gmi-demo-pets"
GMI_API_KEY = os.environ.get("GMI_API_KEY")
HF_DATASET =  "Multimodal-Fatima/CUB_train" #"Multimodal-Fatima/OxfordPets_train" #"Binaryy/multimodal-real-estate-search"
DATASET_SPLIT = "train"
MAX_DATASET_LENGTH = 300 #3000
INGESTION_BATCH_SIZE = 30

if not LANCEDB_API_KEY:
    raise ValueError("LANCEDB_API_KEY is not set")
if not GMI_API_KEY:
    raise ValueError("GMI_API_KEY is not set")

def is_primitive(value):
    return isinstance(value, (int, float, str, bool, bytes))

def get_dynamic_schema():
    dataset = datasets.load_dataset(HF_DATASET, split=DATASET_SPLIT)
    sample = dataset[0]
    
    type_mapping = {
        'string': str,
        'int64': int,
        'int32': int,
        'int16': int,
        'int8': int,
        'float64': float,
        'float32': float,
        'bool': bool,
        'binary': bytes,
    }
    
    fields = {
        'vector': (Vector(512), ...), 
        'image': (bytes, ...)  
    }
    
    for col, value in sample.items():
        if col not in fields:
            if is_primitive(value):
                python_type = type_mapping.get(str(type(value).__name__).lower(), type(value))
                if python_type not in [int, float, str, bool, bytes]:
                    python_type = str 
                fields[col] = (python_type, ...)

    return create_model('DynamicSchema', 
                       **fields,
                       __base__=LanceModel)

Schema = get_dynamic_schema()

def create_or_open_lancedb_table():
    """
    Create or open a LanceDB table.
    """
    db = lancedb.connect(
        uri=LANCEDB_URI,
        api_key=LANCEDB_API_KEY,
        region="us-east-1"
    )
    try:
        return db.open_table(LANCEDB_TABLE_NAME)
    except Exception:
        return db.create_table(LANCEDB_TABLE_NAME, schema=Schema)
