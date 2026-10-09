from langchain_openai import AzureOpenAIEmbeddings
from langchain_azure_cosmosdb import AzureCosmosDBNoSqlVectorSearch
import os
from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

DATABASE_ID = "medical_db0"
CONTAINER_NAME = "encyclopedia"
cosmos_key = os.environ['COSMOS_PRIMARY_KEY']
COSMOS_ENDPOINT = os.environ['COSMOS_ENDPOINT']


cosmos_client = CosmosClient(COSMOS_ENDPOINT, credential=cosmos_key)

database = cosmos_client.create_database_if_not_exists(id=DATABASE_ID)
print(f"Database created or returned: {database.id}")

vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 1536
        }
    ]
}

indexing_policy = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [
        {"path": "/embedding", "type": "quantizedFlat"}
    ]
}


print(f"Ensuring container '{CONTAINER_NAME}' exists with vector policies...")


# ------------------------------------------------------------------
# 1. Configuration & Credentials
# ------------------------------------------------------------------
COSMOS_ENDPOINT = os.environ["COSMOS_ENDPOINT"]
AZURE_EMBEDDING_ENDPOINT = os.environ["AZURE_EMBEDDING_ENDPOINT"]
AZURE_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]

CONTAINER_NAME = "encyclopedia"

database = cosmos_client.get_database_client(DATABASE_ID)
# ------------------------------------------------------------------
# 4. Initialize Embeddings & Cosmos DB Client
# ------------------------------------------------------------------
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",
    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
    api_version="2024-10-21",
    api_key=AZURE_API_KEY,
)

# ------------------------------------------------------------------
# 5. Define Cosmos DB Vector & Indexing Policies
# ------------------------------------------------------------------
# text-embedding-3-small outputs 1536-dimensional vectors
vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 1536
        }
    ]
}

indexing_policy = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": '/"_etag"/?'}],
    "vectorIndexes": [
        {"path": "/embedding", "type": "quantizedFlat"}
    ]
}


print("Database and container verified.")

vector_store = AzureCosmosDBNoSqlVectorSearch(
    cosmos_client=cosmos_client,
    database_name=DATABASE_ID,
    container_name=CONTAINER_NAME,
    embedding=embeddings,
    vector_embedding_policy=vector_embedding_policy,
    indexing_policy=indexing_policy,
    cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
    cosmos_database_properties={"id": DATABASE_ID},
    vector_search_fields={"text_field": "text",
                          "embedding_field": "embedding"},
)

print("Vector store initialized and ready for use.")
