from pinecone_db import PineconeDB

# Initialize
# db = PineconeDB(index_name="langtech-langchain")
# db2 = PineconeDB(index_name="langtech-langgraph")


# Set a default namespace
# db.set_namespace("laptops")

# Example records (metadata only for now; embeddings to be added later)
# records = [
#     {"id": "laptop_1", "values": [0.1, 0.2, ...], "metadata": {"chunk_text": "Sample laptop", "category": "laptop"}},  # Note: 'id' and 'values' for upsert
#     {"id": "laptop_2", "values": [0.3, 0.4, ...], "metadata": {"chunk_text": "Another laptop", "category": "laptop"}}
# ]

# Append to current namespace (laptops)
# db.append_records(records=records)

# Append to a different namespace overriding current
# db.append_records(records=records, namespace="phones")

# Get stats for a namespace
# stats = db.get_index_stats(namespace="laptops")


def append_sample_records():
    """Append sample records to langtech-langchain and langtech-langgraph Pinecone indexes"""
    
    # Initialize the two databases
    try:
        db_langchain = PineconeDB(index_name="langtech-langchain")
        db_langgraph = PineconeDB(index_name="langtech-langgraph")
    except Exception as e:
        print(f"Error initializing PineconeDB: {e}")
        return

    # Sample records for langtech-langchain (e.g., LangChain-related data)
    langchain_records = [
        {
            "_id": "langchain_1",
            "chunk_text": "LangChain is a framework for building applications with LLMs, enabling context-aware and reasoning capabilities.",
            "category": "framework",
            "title": "LangChain Overview"
        },
        {
            "_id": "langchain_2",
            "chunk_text": "LangChain supports tools like vector stores, agents, and memory for conversational AI.",
            "category": "framework",
            "title": "LangChain Features"
        }
    ]

    # Sample records for langtech-langgraph (e.g., LangGraph-related data)
    langgraph_records = [
        {
            "_id": "langgraph_1",
            "chunk_text": "LangGraph is a library for building stateful, graph-based LLM applications with dynamic workflows.",
            "category": "library",
            "title": "LangGraph Introduction"
        },
        {
            "_id": "langgraph_2",
            "chunk_text": "LangGraph enables complex agentic workflows with cycles and branching for advanced AI systems.",
            "category": "library",
            "title": "LangGraph Workflows"
        }
    ]

    # Set namespaces for organization
    db_langchain.set_namespace("langchain-data")
    db_langgraph.set_namespace("langgraph-data")

    # Append records to langtech-langchain
    try:
        success = db_langchain.append_records(records=langchain_records, namespace="langchain-data")
        if success:
            print(f"Successfully appended {len(langchain_records)} records to langtech-langchain")
            db_langchain.get_index_stats(namespace="langchain-data")
        else:
            print("Failed to append records to langtech-langchain")
    except Exception as e:
        print(f"Error appending to langtech-langchain: {e}")

    # Append records to langtech-langgraph
    try:
        success = db_langgraph.append_records(records=langgraph_records, namespace="langgraph-data")
        if success:
            print(f"Successfully appended {len(langgraph_records)} records to langtech-langgraph")
            db_langgraph.get_index_stats(namespace="langgraph-data")
        else:
            print("Failed to append records to langtech-langgraph")
    except Exception as e:
        print(f"Error appending to langtech-langgraph: {e}")

if __name__ == "__main__":
    append_sample_records()