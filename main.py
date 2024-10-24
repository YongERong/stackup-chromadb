__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import os
from dotenv import load_dotenv

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

load_dotenv()

start_time = time.time()
## Initialise ChromaDB and embedding model
client = chromadb.Client()

tags = client.create_collection("tag_collection")
events = client.create_collection("events_collection")
users = client.create_collection("users_collection")


model = SentenceTransformer("all-MiniLM-L6-v2")


## Populate tags collection
tags_df = pd.read_csv("tags.csv", delimiter=":")
tags_list = tags_df.to_csv(None, sep=":", header=False, index=False).split("\n")
tags_list = list(filter(bool, tags_list))
tags_embeddings = model.encode(tags_list)
tags.add(
    embeddings=tags_embeddings,
    metadatas=[{"text": text} for text in tags_list],
    ids=[str(i) for i in range(len(tags_list))],
)


## Create function to tag events by comparing to tag embeddings
def get_tag(event_embedding):
    results = tags.query(query_embeddings=event_embedding, n_results=1)
    return results["metadatas"][0][0]["text"].split(":")[
        0
    ]  # TODO: Fix this janky unnesting


## Populate events collection
events_df = pd.read_csv("events.csv")
events_df["event2embed"] = (
    events_df["name"] + ":" + events_df["description"].astype(str)
)  # Scraper was unable to extract description for all of the events
events_df["event_embeddings"] = model.encode(events_df["event2embed"])
events.add(
    embeddings=events_df["event_embeddings"],
    metadatas=[
        {
            "name": row["name"],
            "link": row["link"],
            "tag": get_tag(row["event_embeddings"]),
            "description": row["description"],
        }
        for _, row in events_df.iterrows()
    ],
    ids=[str(i) for i in range(len(events_df))],
)

end_time = time.time()
db_setup_time = end_time - start_time

## Create tool to retrieve events from events_collection
class GetEvents(BaseModel):
    input: str = Field(
        ...,
        description="The interests and passions of the user based on the conversation.",
    )


def get_events(input: str) -> list[dict]:
    """
    Gets events relevant to the user.
    Args:
        input: A string of the interests and passions of the user
    Returns:
        list[dict]: A list of up to five dictionaries of events related to the user's interests
    """
    interest_embedding = model.encode(input)
    results = events.query(query_embeddings=interest_embedding, n_results=5)

    return results["metadatas"][0] # May need to neaten the output using markdown


get_events_tool = StructuredTool.from_function(
    func=get_events,
    name="getEvents",
    description="Used to get suggest events to the user",
    args_schema=GetEvents,
    return_direct=True,
)

## TODO: Load LLM and bind tool





