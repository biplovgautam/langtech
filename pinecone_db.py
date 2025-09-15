from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import time
import requests
from typing import List, Dict, Any

class PineconeDB:
    """Class to handle Pinecone vector database initialization and record management with Hugging Face embeddings"""

    def __init__(self, index_name: str = "langtech"):
        """
        Initialize Pinecone with the specified index name and Hugging Face API for embeddings.

        Args:
            index_name (str): Name of the Pinecone index (default: 'langtech')
        """
        # Load environment variables
        load_dotenv()

        # Check for API keys
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.hf_api_key = os.environ.get("HF_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        if not self.hf_api_key:
            raise ValueError("HF_API_KEY not found in environment variables")

        # Hugging Face API endpoint for embeddings
        self.hf_api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.hf_headers = {"Authorization": f"Bearer {self.hf_api_key}"}

        # Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.index_name = index_name
            self.namespace = None  # Default namespace is root; can be set via set_namespace()

            # Create index if it doesn't exist
            if not self.pc.has_index(index_name):
                self.pc.create_index(
                    name=index_name,
                    dimension=384,  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                print(f"Created Pinecone index: {index_name}")
                time.sleep(10)  # Wait for index creation to complete

            # Connect to the index
            self.index = self.pc.Index(index_name)
            print(f"Connected to Pinecone index: {index_name}")

        except Exception as e:
            raise Exception(f"Failed to initialize Pinecone: {e}")

    def set_namespace(self, namespace: str):
        """
        Set the namespace for operations. This allows handling different logical collections
        within the same index (e.g., for different data sources or categories).

        Args:
            namespace (str): The namespace to use for subsequent operations
        """
        self.namespace = namespace
        print(f"Set namespace to: {namespace}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text using Hugging Face API.

        Args:
            text (str): Input text to embed

        Returns:
            List[float]: Embedding vector (384-dimensional for all-MiniLM-L6-v2)

        Raises:
            Exception: If the API request fails
        """
        try:
            response = requests.post(self.hf_api_url, headers=self.hf_headers, json={"inputs": text})
            response.raise_for_status()
            embedding = response.json()
            if isinstance(embedding, list) and len(embedding) > 0:
                return embedding[0]  # all-MiniLM-L6-v2 returns list of lists
            else:
                raise ValueError("Unexpected embedding format from Hugging Face API")
        except Exception as e:
            raise Exception(f"Failed to get embedding: {e}")

    def append_records(self, records: List[Dict[str, Any]], namespace: str = None) -> bool:
        """
        Append records to the specified namespace in the Pinecone index, generating embeddings
        for the 'chunk_text' field using Hugging Face API.

        Args:
            records (List[Dict[str, Any]]): List of records to append. Each record must include:
                - '_id': Unique identifier for the record
                - 'chunk_text': Text field to embed
                - Optional: Other metadata fields like 'category', 'title', etc.
            namespace (str, optional): Namespace for this operation. Overrides current namespace if provided.

        Returns:
            bool: True if upsert successful, False otherwise
        """
        try:
            # Use provided namespace or current one, default to None (root)
            target_namespace = namespace if namespace is not None else self.namespace

            # Prepare Pinecone-compatible vectors
            vectors = []
            for record in records:
                if "_id" not in record or "chunk_text" not in record:
                    raise ValueError("Each record must have '_id' and 'chunk_text' fields")
                
                # Generate embedding for chunk_text
                embedding = self.get_embedding(record["chunk_text"])
                
                # Prepare vector format for Pinecone
                vector = {
                    "id": record["_id"],
                    "values": embedding,
                    "metadata": {
                        "chunk_text": record["chunk_text"],
                        **{k: v for k, v in record.items() if k not in ["_id", "chunk_text"]}
                    }
                }
                vectors.append(vector)

            # Upsert vectors to Pinecone
            self.index.upsert(vectors=vectors, namespace=target_namespace)
            print(f"Successfully appended {len(vectors)} records to namespace '{target_namespace}' in index '{self.index_name}'")
            return True

        except Exception as e:
            print(f"Error appending records to Pinecone: {e}")
            return False

    def get_index_stats(self, namespace: str = None) -> Dict[str, Any]:
        """
        Get statistics for the index or a specific namespace.

        Args:
            namespace (str, optional): Namespace to query stats for. Uses current if None.

        Returns:
            Dict[str, Any]: Index statistics
        """
        target_namespace = namespace if namespace is not None else self.namespace
        try:
            stats = self.index.describe_index_stats()
            namespace_stats = stats.get("namespaces", {}).get(target_namespace, {})
            print(f"Index stats for namespace '{target_namespace}': {namespace_stats}")
            return namespace_stats
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {}