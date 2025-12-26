"""
Vector database using Chroma
"""

import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory="data/vectordb"
    )
)

collection = client.get_or_create_collection(name="research_papers")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def add_chunks(paper_id, chunks):
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{paper_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"paper_id": paper_id} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )


def retrieve(query, k=5):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )
    return results["documents"][0]
