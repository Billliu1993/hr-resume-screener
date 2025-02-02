import os
import pickle
import json
import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL = "voyage-3-lite"
BATCH_SIZE = 128


class VectorDB:
    def __init__(self, name, api_key=None):
        if api_key is None:
            api_key = os.getenv("VOYAGE_API_KEY")
        self.client = voyageai.Client(api_key=api_key)
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/vector_db.pkl"

    def _create_overlapping_chunks(self, data, overlap_size):
        """Create overlapping chunks from the input data."""
        overlapped_chunks = []
        overlapped_metadata = []

        for i in range(len(data)):
            current_chunk = data[i]
            if i > 0:  # Add overlap with previous chunk
                prev_chunk = data[i - 1]
                words = prev_chunk["text"].split()
                overlap_text = " ".join(words[-overlap_size:])
                current_chunk["text"] = f"{overlap_text} {current_chunk['text']}"

            if i < len(data) - 1:  # Add overlap with next chunk
                next_chunk = data[i + 1]
                words = next_chunk["text"].split()
                overlap_text = " ".join(words[:overlap_size])
                current_chunk["text"] = f"{current_chunk['text']} {overlap_text}"

            overlapped_chunks.append(current_chunk)
            overlapped_metadata.append(
                {
                    "chunk_heading": current_chunk["chunk_heading"],
                    "original_index": i,
                    "has_prev_overlap": i > 0,
                    "has_next_overlap": i < len(data) - 1,
                }
            )

        return overlapped_chunks, overlapped_metadata

    def clear_db(self):
        """Clear all database contents and cache"""
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def load_data(self, data, overlap_words=0, force_reload=False):
        """Load data into vector database with option to force reload
        
        Args:
            data: List of text chunks to embed
            overlap_words: Number of words to overlap between chunks
            force_reload: If True, clear existing DB and reload
        """
        if force_reload:
            self.clear_db()
        
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.") 
            return

        if os.path.exists(self.db_path) and not force_reload:
            print("Loading vector database from disk.")
            self.load_db()
            return

        if overlap_words > 0:
            data, extra_metadata = self._create_overlapping_chunks(data, overlap_words)
            self.metadata.extend(extra_metadata)

        texts = [f"Heading: {item['chunk_heading']}\n\n Chunk Text:{item['text']}" for item in data]
        self._embed_and_store(texts, data)
        self.save_db()
        print("Vector database loaded and saved.")

    def _embed_and_store(self, texts, data):
        batch_size = BATCH_SIZE
        result = [
            self.client.embed(
                texts[i : i + batch_size], model=EMBEDDING_MODEL
            ).embeddings
            for i in range(0, len(texts), batch_size)
        ]
        self.embeddings = [embedding for batch in result for embedding in batch]
        self.metadata = data

    def search(self, query, k=5, similarity_threshold=0.2):
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.client.embed(
                [query], model=EMBEDDING_MODEL
            ).embeddings[0]
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1]
        top_examples = []

        for idx in top_indices:
            if similarities[idx] >= similarity_threshold:
                example = {
                    "metadata": self.metadata[idx],
                    "similarity": similarities[idx],
                }
                top_examples.append(example)

                if len(top_examples) >= k:
                    break
        self.save_db()
        return top_examples

    def save_db(self):
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError(
                "Vector database file not found. Use load_data to create a new database."
            )
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])
