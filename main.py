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
event_embeddings = model.encode(events_df["event2embed"])
events.add(
    embeddings=event_embeddings,
    metadatas=[
        {
            "name": row["name"],
            "link": row["link"],
            "tag": get_tag(event_embeddings[i]),
            "description": row["description"],
        }
        for i, row in events_df.iterrows()
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


import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    
    # other params...
)


memory = MemorySaver()
app = create_react_agent(
    llm,
    tools=[get_events_tool],
    checkpointer=memory,
    messages_modifier=SystemMessage(
        """Your name is Passion Weaver, you are a chatbot which helps youth to identify their interests and find their passions by speaking to them and suggesting relevant events to them.
        Start by asking the three of the five following questions one at a time:
        1. What activities or hobbies do you find yourself losing track of time while doing?
        2. If you had a whole day free with no obligations, what would you most want to spend your time doing?
        3. What kind of topics do you enjoy learning or talking about, even outside of school or work?
        4. Are there any problems or causes in the world that you feel strongly about?
        5. What skills or talents do people often compliment you on, or have you noticed you excel at?
        
        After that, use the get_events_tool to get events relevant to the user, and continue the conversation if necessary to gain a better understanding of the user""")
)

thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}


def chatbot(message:str) -> str:
    return app.invoke({"messages": [HumanMessage(message)]}, config, stream_mode="values")["messages"][-1].content


while True:
    print(chatbot(input("> ")))





