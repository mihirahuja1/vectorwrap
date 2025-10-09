# vectorwrap Integrations

vectorwrap provides seamless integration with popular vector databases and AI frameworks.

## LangChain Integration

Use vectorwrap backends with LangChain's document retrieval and RAG pipelines.

### Installation

```bash
pip install "vectorwrap[langchain]"
```

### Usage

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from vectorwrap.integrations.langchain import VectorwrapStore

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Create vector store (supports any vectorwrap backend)
vectorstore = VectorwrapStore(
    connection_url="postgresql://user:pass@localhost/db",
    collection_name="documents",
    embedding_function=embeddings,
    dimension=1536
)

# Add documents
texts = ["Hello world", "LangChain is great for RAG"]
metadatas = [{"source": "intro.txt"}, {"source": "review.txt"}]
vectorstore.add_texts(texts, metadatas)

# Search
results = vectorstore.similarity_search("greeting", k=5)
for doc in results:
    print(doc.page_content, doc.metadata)

# Use as retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
docs = retriever.get_relevant_documents("What is LangChain?")
```

### Quick Start

```python
from langchain.embeddings import OpenAIEmbeddings
from vectorwrap.integrations.langchain import VectorwrapStore

embeddings = OpenAIEmbeddings()

# SQLite for prototyping
store = VectorwrapStore.from_texts(
    texts=["doc1", "doc2"],
    embedding=embeddings,
    connection_url="sqlite:///:memory:"
)

# PostgreSQL for production
store = VectorwrapStore.from_texts(
    texts=["doc1", "doc2"],
    embedding=embeddings,
    connection_url="postgresql://user:pass@localhost/db"
)
```

## LlamaIndex Integration

Use vectorwrap with LlamaIndex's data framework and query engines.

### Installation

```bash
pip install "vectorwrap[llamaindex]"
```

### Usage

```python
from llama_index.core import VectorStoreIndex, Document, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from vectorwrap.integrations.llamaindex import VectorwrapVectorStore

# Create vector store
vector_store = VectorwrapVectorStore(
    connection_url="postgresql://user:pass@localhost/db",
    collection_name="documents",
    dimension=1536
)

# Create index
embed_model = OpenAIEmbedding()
service_context = ServiceContext.from_defaults(embed_model=embed_model)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    service_context=service_context
)

# Add documents
documents = [
    Document(text="Hello world"),
    Document(text="LlamaIndex is powerful")
]
index.insert_nodes(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is this about?")
print(response)
```

### Convenience Function

```python
from vectorwrap.integrations.llamaindex import create_vector_store

# SQLite
store = create_vector_store("sqlite", path=":memory:")

# PostgreSQL
store = create_vector_store(
    "postgres",
    host="localhost",
    user="user",
    password="pass",
    database="vectordb"
)

# ClickHouse
store = create_vector_store(
    "clickhouse",
    host="localhost",
    user="default",
    database="default"
)
```

## Supabase Integration

Simplified interface for Supabase's managed PostgreSQL + pgvector.

### Installation

```bash
pip install vectorwrap  # Core package includes PostgreSQL support
```

### Usage

```python
from vectorwrap.integrations.supabase import SupabaseVectorStore

# From credentials
store = SupabaseVectorStore.from_supabase_credentials(
    project_url="https://xxx.supabase.co",
    service_key="your-service-key"
)

# From environment variables
# Set: SUPABASE_URL and SUPABASE_SERVICE_KEY
store = SupabaseVectorStore.from_env()

# Create collection with RLS
store.create_collection("documents", dim=1536, enable_rls=True)

# Bulk upsert
vectors = [[0.1, 0.2, ...] for _ in range(100)]
metadatas = [{"source": f"doc{i}"} for i in range(100)]
store.bulk_upsert("documents", vectors=vectors, metadatas=metadatas)

# Query
results = store.query("documents", query_vector, top_k=10)
```

### RLS Policies

```python
# Create Row Level Security policy
store.create_rls_policy(
    "documents",
    "user_documents",
    "auth.uid() = (metadata->>'user_id')::uuid"
)
```

### Schema Export

```python
# Generate SQL for Supabase SQL Editor
sql = store.get_schema_sql("documents", dim=1536)
print(sql)
```

### Migration

```python
from vectorwrap.integrations.supabase import migrate_from_pinecone
import pinecone

# Setup
pinecone.init(api_key="...", environment="...")
index = pinecone.Index("my-index")

store = SupabaseVectorStore.from_env()

# Migrate from Pinecone
migrate_from_pinecone(index, store, "documents")
```

## Milvus Integration

Use vectorwrap-compatible interface with Milvus vector database.

### Installation

```bash
pip install "vectorwrap[milvus]"
```

### Usage

```python
from vectorwrap.integrations.milvus import MilvusBackend

# Connect to Milvus
db = MilvusBackend(
    host="localhost",
    port="19530"
)

# Create collection with HNSW index
db.create_collection(
    "documents",
    dim=1536,
    index_type="HNSW",
    metric_type="L2"
)

# Upsert vectors
db.upsert("documents", 1, [0.1, 0.2, ...], {"source": "doc1"})

# Query with filters
results = db.query(
    "documents",
    query_vector,
    top_k=10,
    filter={"source": "doc1"}
)

# Collection stats
stats = db.get_collection_stats("documents")
print(stats)
```

### Export to Milvus

```python
from vectorwrap.integrations.milvus import export_to_milvus, MilvusBackend

# Prepare data
vectors = [[0.1, 0.2, ...] for _ in range(1000)]
metadatas = [{"id": i} for i in range(1000)]

# Export
milvus = MilvusBackend(host="localhost", port="19530")
export_to_milvus(
    collection_name="documents",
    vectors=vectors,
    metadatas=metadatas,
    milvus_backend=milvus,
    dim=1536
)
```

## Qdrant Integration

Use vectorwrap-compatible interface with Qdrant vector database.

### Installation

```bash
pip install "vectorwrap[qdrant]"
```

### Usage

```python
from vectorwrap.integrations.qdrant import QdrantBackend

# Local Qdrant
db = QdrantBackend(url="http://localhost:6333")

# Qdrant Cloud
db = QdrantBackend(
    url="https://xxx.cloud.qdrant.io",
    api_key="your-api-key"
)

# Create collection
db.create_collection("documents", dim=1536, distance="Cosine")

# Upsert vectors
db.upsert("documents", 1, [0.1, 0.2, ...], {"source": "doc1"})

# Query with filters and score threshold
results = db.query(
    "documents",
    query_vector,
    top_k=10,
    filter={"source": "doc1"},
    score_threshold=0.8
)

# Scroll collection (for export)
points = db.scroll_collection("documents", limit=1000, with_payload=True)
```

### URL Convenience

```python
from vectorwrap.integrations.qdrant import create_qdrant_from_url

# Local
db = create_qdrant_from_url("qdrant://localhost:6333")

# Cloud
db = create_qdrant_from_url(
    "qdrant+cloud://xxx.cloud.qdrant.io?api_key=your-key"
)
```

### Export to Qdrant

```python
from vectorwrap.integrations.qdrant import export_to_qdrant, QdrantBackend

# Prepare data
ids = list(range(1000))
vectors = [[0.1, 0.2, ...] for _ in range(1000)]
metadatas = [{"id": i} for i in range(1000)]

# Export
qdrant = QdrantBackend(url="http://localhost:6333")
export_to_qdrant(
    collection_name="documents",
    ids=ids,
    vectors=vectors,
    metadatas=metadatas,
    qdrant_backend=qdrant,
    dim=1536
)
```

## Comparison Table

| Integration | Use Case | Install | Production Ready |
|-------------|----------|---------|------------------|
| **LangChain** | RAG pipelines, document retrieval | `vectorwrap[langchain]` | Yes |
| **LlamaIndex** | Data frameworks, query engines | `vectorwrap[llamaindex]` | Yes |
| **Supabase** | Managed PostgreSQL, Auth + vectors | `vectorwrap` | Yes |
| **Milvus** | Large-scale vector search | `vectorwrap[milvus]` | Yes |
| **Qdrant** | Cloud-native vector search | `vectorwrap[qdrant]` | Yes |

## Migration Paths

### From Pinecone to Supabase

```python
from vectorwrap.integrations.supabase import migrate_from_pinecone
import pinecone

pinecone.init(api_key="...", environment="...")
index = pinecone.Index("my-index")

store = SupabaseVectorStore.from_env()
migrate_from_pinecone(index, store, "documents")
```

### From PostgreSQL to Milvus

```python
from vectorwrap.integrations.milvus import MilvusBackend, migrate_from_pgvector

milvus = MilvusBackend(host="localhost", port="19530")

migrate_from_pgvector(
    "postgresql://user:pass@localhost/db",
    "documents",
    milvus,
    "documents"
)
```

### From SQL to Qdrant

```python
from vectorwrap.integrations.qdrant import QdrantBackend, migrate_from_sql_store

qdrant = QdrantBackend(url="http://localhost:6333")

migrate_from_sql_store(
    "postgresql://user:pass@localhost/db",
    "documents",
    qdrant,
    "documents"
)
```

## Installation Options

```bash
# All integrations
pip install "vectorwrap[all]"

# Specific integrations
pip install "vectorwrap[langchain]"
pip install "vectorwrap[llamaindex]"
pip install "vectorwrap[milvus]"
pip install "vectorwrap[qdrant]"

# Multiple integrations
pip install "vectorwrap[langchain,llamaindex]"

# All integrations only (no extra databases)
pip install "vectorwrap[integrations]"
```

## Notes

- All integrations maintain vectorwrap's simple API
- Switching between backends requires only changing the connection URL
- Metadata filtering support varies by backend
- For production use, see individual backend documentation for optimization tips
