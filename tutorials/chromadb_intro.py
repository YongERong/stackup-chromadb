__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
client = chromadb.Client()

collection = client.create_collection("simple_collection")

ids = ["1","2"]

embeddings = [
  [1.0, 0.0, 0.0],
  [0.0, 1.0, 0.0]]

metadata = [
  {"info":"This is the first item."},
  {"info":"This is the second item."}]

collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadata)

query_embedding = [0.9, 0.1, 0.0]
results = collection.query(query_embeddings=[query_embedding], n_results=1)

print(results["metadatas"])
