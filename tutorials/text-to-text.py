__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from sentence_transformers import SentenceTransformer
import chromadb

client = chromadb.Client()
collection = client.create_collection("text_collection")

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [
    "The Eiffel Tower in Paris is an iconic landmark.",
    "A delicious homemade pizza fresh from the oven.",
    "A scientist studying the effects of climate change on polar bears.",
    "A group of friends hiking through the mountains during autumn.",
    "The discovery of water on Mars could mean the possibility of life.",
    "The rise of artificial intelligence is reshaping industries.",
    "A peaceful garden filled with blooming flowers and chirping birds.",
    "A musician playing the violin in a crowded subway station.",
    "A historical novel set during World War II.",
    "An astronaut floating weightlessly in space.",
     "The space shuttle launched into orbit, marking a new era in space exploration.",
    "An advanced AI system is helping doctors diagnose rare diseases.",
    "A cozy cafe in the city center where people gather to relax and chat."]

embeddings = model.encode(texts)

collection.add(
    embeddings=embeddings,
    metadatas=[{"text": text} for text in texts],
    ids=[str(i) for i in range(len(texts))]
)

queries = [
    "A musician performing in public",
    "A peaceful place with flowers and birds",
    "Medical professionals using machine learning to find rare illnesses", 
    "Astronaughts traveling to outside of earth",
    "Relaxing at a shop with friends",
]

for query in queries:
    print(f"\nQuery: {query}")
    
    # Generate an embedding for the query text
    query_embedding = model.encode(query)

    # Perform a vector search in the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # Retrieve top 3 similar entries
    )

    for i, result in enumerate(results['metadatas'][0]):
         print(f"Result {i + 1}: {result['text']}")\
         

# OTHER MODELS TO TRY:
# all-MiniLM-L6-v2
# paraphrase-distilroberta-base-v1
# msmarco-distilbert-base-v4 
# paraphrase-MiniLM-L12-v2
# cross-encoder/ms-marco-TinyBERT-L6
