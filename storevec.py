from pinecone_db import PineconeDB

# Initialize
# db = PineconeDB(index_name="langtech")

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