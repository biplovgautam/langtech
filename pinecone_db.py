from pinecone import Pinecone
from dotenv import load_dotenv
import os
import time
from typing import List, Dict, Any

class PineconeDB:
    """Class to handle Pinecone vector database initialization and record management"""

    def __init__(self, index_name: str = "langtech"):
        """
        Initialize Pinecone with the specified index name.

        Args:
            index_name (str): Name of the Pinecone index (default: 'langtech')
        """
        # Load environment variables
        load_dotenv()

        # Check for API key
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        # Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=api_key)
            self.index_name = index_name
            self.namespace = None  # Default namespace is root; can be set via set_namespace()

            # Create index if it doesn't exist
            if not self.pc.has_index(index_name):
                self.pc.create_index_for_model(
                    name=index_name,
                    cloud="aws",
                    region="us-east-1",
                    embed={
                        "model": "llama-text-embed-v2",  # Placeholder embedding model; can be updated later
                        "field_map": {"text": "chunk_text"}
                    }
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

    def append_records(self, records: List[Dict[str, Any]], namespace: str = None) -> bool:
        """
        Append records to the specified namespace in the Pinecone index.
        If no namespace is provided, uses the currently set namespace or root if none set.

        Args:
            records (List[Dict[str, Any]]): List of records to append. Each record must include:
                - '_id': Unique identifier for the record
                - 'chunk_text': Text field for embedding (will be embedded later)
                - Optional: Other metadata fields like 'category', 'title', etc.
            namespace (str, optional): Namespace for this operation. Overrides current namespace if provided.

        Returns:
            bool: True if upsert successful, False otherwise

        Note: Embeddings are not generated here. Records should have pre-computed 'values' key
        with vector embeddings when embeddings are implemented. For now, this upserts metadata
        records without vectors (index will store them as zero-vectors or require embedding step).
        """
        try:
            # Use provided namespace or current one, default to None (root)
            target_namespace = namespace if namespace is not None else self.namespace
            
            # Upsert records to the specified namespace
            self.index.upsert(vectors=records, namespace=target_namespace)
            record_count = len(records)
            print(f"Successfully appended {record_count} records to namespace '{target_namespace}' in index '{self.index_name}'")
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
            stats = self.index.describe_index_stats(namespace=target_namespace)
            print(f"Index stats for namespace '{target_namespace}': {stats}")
            return stats
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {}